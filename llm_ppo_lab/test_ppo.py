"""Small CPU checks of token-level PPO semantics, with no pretrained download."""
from contextlib import nullcontext
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

import torch

from core import (response_mask, build_rewards, finite_episode_gae, ppo_loss,
                  token_statistics, masked_mean)
from models import Bundle, TinyActorCritic, HFActorCritic
from train import collect, update, update_directory, make_data


class TokenPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_first_eos_included_even_when_eos_is_padding(self):
        ids = torch.tensor([[7,3,3,3],[3,3,3,3],[7,8,9,10]])
        expected = torch.tensor([[1,1,0,0],[1,0,0,0],[1,1,1,1]], dtype=torch.bool)
        torch.testing.assert_close(response_mask(ids,3),expected)

    def test_terminal_score_once_and_sampled_kl_can_have_either_sign(self):
        mask = torch.tensor([[1,1,0],[1,0,0]],dtype=torch.bool)
        old = torch.tensor([[-.2,-.3,5.],[-.8,5.,5.]],requires_grad=True)
        ref = torch.tensor([[-.4,-.2,0.],[-.6,0.,0.]],requires_grad=True)
        reward,kl = build_rewards(torch.tensor([1.,2.]),old,ref,mask,.5)
        torch.testing.assert_close(kl,torch.tensor([[.2,-.1,0.],[-.2,0.,0.]]))
        torch.testing.assert_close(reward,torch.tensor([[-.1,1.05,0.],[2.1,0.,0.]]))
        no_kl,_ = build_rewards(torch.tensor([1.,2.]),old,old,mask,.5)
        torch.testing.assert_close(no_kl,torch.tensor([[0.,1.,0.],[2.,0.,0.]]))
        self.assertFalse(reward.requires_grad);self.assertFalse(kl.requires_grad)

    def test_variable_length_returns_and_final_bootstrap_zero(self):
        mask=torch.tensor([[1,1,1],[1,0,0]],dtype=torch.bool)
        reward=torch.tensor([[0.,0.,4.],[2.,0.,0.]])
        values=torch.tensor([[1.,2.,3.],[9.,99.,99.]],requires_grad=True)
        advantage,returns=finite_episode_gae(reward,values,mask,gamma=1.,lam=1.)
        torch.testing.assert_close(returns,torch.tensor([[4.,4.,4.],[2.,0.,0.]]))
        torch.testing.assert_close(advantage,torch.tensor([[3.,2.,1.],[-7.,0.,0.]]))
        self.assertFalse(returns.requires_grad)

    def test_value_is_before_action_and_logits_are_causal(self):
        torch.manual_seed(4);model=TinyActorCritic(width=16).eval()
        prompt=torch.tensor([[1,4,5,2]]);attention=torch.ones_like(prompt)
        first=torch.tensor([[6,7,3]]);second=torch.tensor([[9,7,3]])
        a,va,_=token_statistics(model,prompt,attention,first,torch.ones_like(first))
        b,vb,_=token_statistics(model,prompt,attention,second,torch.ones_like(second))
        # State value for response action 0 must not see which action was picked.
        torch.testing.assert_close(va[:,0],vb[:,0])
        full_logits,_=model(torch.cat([prompt,first],1),torch.ones(1,7,dtype=torch.long))
        expected=full_logits[:,3:6].log_softmax(-1).gather(-1,first[...,None]).squeeze(-1)
        torch.testing.assert_close(a,expected)
        # Altering future response tokens cannot alter first-step distribution.
        changed=first.clone();changed[:,1]=10
        _,vc,_=token_statistics(model,prompt,attention,changed,torch.ones_like(first))
        torch.testing.assert_close(va[:,:2],vc[:,:2])

    def test_masks_and_frozen_old_statistics_do_not_receive_gradient(self):
        logp=torch.tensor([[-.5,7.]],requires_grad=True)
        value=torch.tensor([[.2,9.]],requires_grad=True)
        old=logp.detach().clone().requires_grad_(True)
        old_value=value.detach().clone().requires_grad_(True)
        advantage=torch.tensor([[1.,900.]],requires_grad=True)
        target=torch.tensor([[1.,900.]],requires_grad=True)
        loss,_=ppo_loss(logp,value,torch.zeros_like(logp),old,old_value,advantage,target,
                        torch.tensor([[1.,0.]]),value_coef=.5)
        loss.backward()
        self.assertIsNone(old.grad);self.assertIsNone(old_value.grad)
        self.assertIsNone(advantage.grad);self.assertIsNone(target.grad)
        self.assertEqual(float(logp.grad[0,1]),0.);self.assertEqual(float(value.grad[0,1]),0.)

    def test_valid_token_mean_is_not_silent_per_sequence_mean(self):
        value=torch.tensor([[1.,1.,1.],[9.,999.,999.]])
        mask=torch.tensor([[1.,1.,1.],[1.,0.,0.]])
        self.assertEqual(float(masked_mean(value,mask)),3.)

    def test_collect_and_update_preserve_old_policy_and_fixed_reference(self):
        torch.manual_seed(2);bundle=Bundle('tiny','unused','unused','cpu')
        reference={key:value.clone() for key,value in bundle.reference.state_dict().items()}
        rows=[dict(digits='12',answer='21',prompt='unused'),dict(digits='34',answer='43',prompt='unused')]
        batch,_=collect(bundle,rows,'cpu',4,.05,1.,.95)
        old=batch['old_logp'].clone()
        optimizer=torch.optim.Adam(bundle.policy.parameters(),lr=.001)
        update(bundle,optimizer,batch,epochs=2,clip=.2,value_coef=.5,entropy_coef=0.,normalize=True)
        torch.testing.assert_close(old,batch['old_logp'],rtol=0,atol=0)
        for key,value in bundle.reference.state_dict().items():
            torch.testing.assert_close(value,reference[key],rtol=0,atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in bundle.reference.parameters()))

    def test_hf_lora_reference_stays_frozen_without_download(self):
        try:
            from transformers import Qwen2Config,Qwen2ForCausalLM
            import peft
        except ImportError:
            self.skipTest('Optional transformers/peft dependencies are absent')
        config=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
                          num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1)
        with tempfile.TemporaryDirectory() as tmp:
            Qwen2ForCausalLM(config).save_pretrained(tmp)
            model=HFActorCritic(tmp,None,'cpu',local_only=True).eval()
            ids=torch.tensor([[1,4,5,6]]);mask=torch.ones_like(ids)
            with torch.no_grad(),model.lm.disable_adapter():before=model(ids,mask)[0].clone()
            trainable_before={name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad}
            logits,value=model(ids,mask)
            loss=-logits[:,-1].log_softmax(-1)[0,7]+value.square().mean()
            loss.backward()
            torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.05).step()
            with torch.no_grad(),model.lm.disable_adapter():after=model(ids,mask)[0]
            torch.testing.assert_close(before,after,rtol=0,atol=0)
            self.assertTrue(all(p.grad is None for name,p in model.lm.named_parameters() if 'lora_' not in name))
            self.assertTrue(any(not torch.equal(p.detach(),trainable_before[name])
                                for name,p in model.named_parameters() if 'lora_' in name))
            self.assertFalse(torch.equal(model.value_head.weight.detach(),trainable_before['value_head.weight']))
            restored=HFActorCritic(tmp,None,'cpu',local_only=True).eval()
            restored.load_trainable_state(model.trainable_state())
            with torch.no_grad():
                expected_logits,expected_values=model(ids,mask)
                actual_logits,actual_values=restored(ids,mask)
            torch.testing.assert_close(expected_logits,actual_logits,rtol=0,atol=0)
            torch.testing.assert_close(expected_values,actual_values,rtol=0,atol=0)
            broken=model.trainable_state();broken['adapter']=dict(broken['adapter'])
            broken['adapter'].pop(next(iter(broken['adapter'])))
            with self.assertRaisesRegex(ValueError,'Incomplete adapter'):
                restored.load_trainable_state(broken)

            # Different prompt lengths, left padding, different EOS positions.
            prompt=torch.tensor([[0,0,1,5],[1,4,5,6]])
            prompt_mask=torch.tensor([[0,0,1,1],[1,1,1,1]])
            response=torch.tensor([[7,3,3],[8,9,3]])  # EOS==3; PAD==EOS for response
            valid=response_mask(response,3)
            sequence=prompt.clone();attention=prompt_mask.clone();online_logp=[]
            with torch.no_grad():
                for t in range(3):
                    logits,_=model(sequence,attention)
                    online_logp.append(logits[:,-1].log_softmax(-1).gather(-1,response[:,t,None]).squeeze(-1))
                    sequence=torch.cat([sequence,response[:,t,None]],-1)
                    attention=torch.cat([attention,valid[:,t,None].long()],-1)
                recomputed,_,_=token_statistics(model,prompt,prompt_mask,response,valid)
            torch.testing.assert_close(recomputed[valid],torch.stack(online_logp,1)[valid],atol=2e-6,rtol=2e-6)

    def test_partial_attempt_is_preserved_before_checkpoint_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);first=update_directory(root,3)
            (first/'rollout.json').write_text('original attempt')
            retry=update_directory(root,3)
            self.assertEqual(retry.name,'update_0003')
            archived=list(root.glob('update_0003.interrupted_*'))
            self.assertEqual(len(archived),1)
            self.assertEqual((archived[0]/'rollout.json').read_text(),'original attempt')
            self.assertTrue((archived[0]/'interruption.json').is_file())

    def test_hf_manifest_does_not_reuse_old_grpo_evaluations_for_training(self):
        data=make_data('hf')
        partitions=[{row['digits'] for row in data[key]} for key in ('train','validation','test')]
        self.assertEqual([len(partition) for partition in partitions],[1024,128,256])
        self.assertFalse(partitions[0]&partitions[1] or partitions[0]&partitions[2] or partitions[1]&partitions[2])
        excluded=set(data['excluded_prior_grpo_evaluation_digits'])
        self.assertEqual(len(excluded),512)
        self.assertFalse(excluded&set.union(*partitions))

    def test_real_tiny_cli_checkpoint_resume_and_inference(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'
            environment=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            command=[sys.executable,str(root/'llm_ppo_lab/train.py'),'--backend','tiny',
                '--device','cpu','--updates','1','--epochs','1','--batch-size','2',
                '--eval-n','2','--out',str(out)]
            for args in (command,command+['--resume']):
                result=subprocess.run(args,cwd=root,env=environment,capture_output=True,text=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            saved=torch.load(out/'latest.pt',weights_only=True,map_location='cpu')
            before=torch.load(out/'before.pt',weights_only=True,map_location='cpu')
            self.assertEqual(saved['update'],2);self.assertEqual(saved['optimizer_steps'],2)
            for key,value in before['reference'].items():
                torch.testing.assert_close(value,saved['reference'][key],rtol=0,atol=0)
            output=Path(tmp)/'inference.json'
            result=subprocess.run([sys.executable,str(root/'llm_ppo_lab/infer.py'),
                '--checkpoint',str(out/'after.pt'),'--out',str(output)],cwd=root,
                env=environment,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            records=json.loads(output.read_text())
            self.assertEqual(records['policy_updates_during_inference'],0)
            self.assertEqual(records['n'],16)
            self.assertIsInstance(saved['config']['torch'],str)


if __name__=='__main__':unittest.main()

/* The game is deterministic. Only playPolicy draws random actions. */
const Breakout = (() => {
  const width = 7;
  const floor = 4;
  const brick = { x: 5, y: 0 };
  const opening = { x: 1, y: 2, vx: 1, vy: 1, paddle: 1, score: 0, tick: 0, outcome: null };

  function step(state, action) {
    if (state.outcome) throw new Error("Reset after the episode ends");
    const paddle = Math.min(width - 1, Math.max(0, state.paddle + action));
    let vx = state.vx;
    if (state.x + vx < 0 || state.x + vx >= width) vx = -vx;
    const x = state.x + vx;
    if (x === 0 || x === width - 1) vx = -vx;
    const y = state.y + state.vy;
    let vy = state.vy;
    let outcome = null;
    let reward = 0;
    if (y === floor) {
      if (x === paddle) vy = -1;
      else outcome = "miss";
    }
    if (y === brick.y) {
      if (x === brick.x) {
        reward = 1;
        outcome = "hit";
      } else {
        vy = 1;
      }
    }
    return {
      state: { x, y, vx, vy, paddle, reward, score: state.score + reward, tick: state.tick + 1, outcome },
      reward,
    };
  }

  function playPolicy(rightProbability, random = Math.random) {
    let state = opening;
    while (!state.outcome) {
      const draw = random();
      // Only the first two moves affect success from this opening.
      const action = draw < rightProbability ? 1 : draw < (1 + rightProbability) / 2 ? 0 : -1;
      state = step(state, action).state;
    }
    return state.score;
  }

  return { width, floor, brick, opening, step, playPolicy };
})();

if (typeof module !== "undefined" && module.exports) module.exports = Breakout;

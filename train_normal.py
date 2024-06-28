import gym
from policy_gradient_agent import PolicyGradientAgent
from tqdm import tqdm
def train(
        env_name: str,
        n_episodes: int = 1000,
        max_t: int = 1000,
        learning_rate: float = 0.01
    ):
    env = gym.make(env_name)
    assert env.observation_space.shape is not None
    assert env.action_space.shape is not None
    agent = PolicyGradientAgent(
        env.observation_space.shape[0],
        env.action_space.n,
        learning_rate
    )
    max_rewards = 0
    for episode in tqdm(range(n_episodes)):
        state, _ = env.reset()
        episode_rewards = 0

        for t in range(max_t):
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_reward(reward)
            episode_rewards += reward
            state = next_state
            if done:
                break
        if episode_rewards > max_rewards:
            max_rewards = episode_rewards
            total_score = 0 
            for _ in range(50):
                state, _ = env.reset()
                for _ in range(500):
                    action = agent.select_action(state)
                    next_state, reward, done, score, _ = env.step(action)
                    state = next_state
                    if done:
                        break
                total_score += score
            print(f"Episode {episode}, Total Reward: {episode_rewards}, Average Score: {total_score / 50}")
            agent.save('policy_snake.pth')
        agent.update_policy()
    
    env.close()
import gym_game
if __name__ == "__main__":
    train(env_name='Snake-v1', n_episodes=5000, max_t=500, learning_rate=0.01)
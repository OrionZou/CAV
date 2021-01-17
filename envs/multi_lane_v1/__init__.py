from gym.envs.registration import register

register(
    id='multi_lane_c-v6',
    entry_point='envs.multi_lane_v1.env_singlecar_gym_c:SUMO_ENV',
    # max_episode_steps=160,
    # reward_threshold=10.0,
)
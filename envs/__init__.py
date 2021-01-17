from gym.envs.registration import register
register(
    id='sumo_env-v6',
    entry_point='envs.naive_v6_1_1.env_singlecar_gym:SUMO_ENV',
    # max_episode_steps=160,
    # reward_threshold=10.0,
)
register(
    id='multi_lane-v6',
    entry_point='envs.naive_v6_1_1.env_singlecar_gym:SUMO_ENV',
    # max_episode_steps=160,
    # reward_threshold=10.0,
)
from gym.envs.registration import register

# register(
#     id='sumo_env_b-v6',
#     entry_point='envs.naive_v6_1_1.env_singlecar_gym_b:SUMO_ENV',
#     # max_episode_steps=160,
#     # reward_threshold=10.0,
# )
# register(
#     id='sumo_env_b_c-v6',
#     entry_point='envs.naive_v6_1_1.env_singlecar_gym_b_c:SUMO_ENV',
#     # max_episode_steps=160,
#     # reward_threshold=10.0,
# )
register(
    id='sumo_env_c-v6',
    entry_point='envs.naive_v6_1_1.env_singlecar_gym:SUMO_ENV',
    # max_episode_steps=160,
    # reward_threshold=10.0,
)
# register(
#     id='sumo_env_spares-v6',
#     entry_point='envs.naive_v6_1_1.env_singlecar_gym_spares:SUMO_ENV',
#     # max_episode_steps=160,
#     # reward_threshold=10.0,
# )
# register(
#     id='sumo_env_spares_c-v6',
#     entry_point='envs.naive_v6_1_1.env_singlecar_gym_spares_c:SUMO_ENV',
#     # max_episode_steps=160,
#     # reward_threshold=10.0,
# )

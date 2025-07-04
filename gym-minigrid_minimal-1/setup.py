from setuptools import setup

setup(
    name='gym_minigrid',
    version='1.0.2',
    keywords='memory, environment, agent, rl, openaigym, openai-gym, gym',
    url='https://github.com/maximecb/gym-minigrid',
    description='Minimalistic gridworld package for OpenAI Gym, reduced to one custom environment',
    packages=['gym_minigrid', 'gym_minigrid.envs'],
    install_requires=[
        'gym==0.17.0',
        'numpy>=1.21.6'
    ]
)

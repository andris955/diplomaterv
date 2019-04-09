from multi_task import MultiTasking
import global_config
import agent
import local_config


if __name__ == '__main__':
    amta = agent.Agent(global_config.MTI1)
    mt = MultiTasking(global_config.MTI1, "A5C", 3, global_config.target_performances, global_config.l, global_config.MaxSteps, amta)
    if local_config.train:
        mt.train()
    else:
        for game in global_config.MTI1:
            amta.play(local_config.model_path, game, 3, True)

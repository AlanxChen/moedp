import hydra
import pathlib
from train import TrainWorkspace
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'moe_dp', 'config'))
)
def main(cfg):
    workspace = TrainWorkspace(cfg)
    workspace.eval()

if __name__ == "__main__":
    main()

from loguru import logger


from util import create_parser, set_global_seed, setup_mlflow, setup_logger, setup_config
from pyg.train import train_gnn
from pyg.inference import infer_gnn

def main():
    
    # Setup arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup configuration
    config = setup_config(vars(args))

    # Setup logger
    setup_logger()
    
    
    # Setup MLflow
    setup_mlflow(config)
        
    # Set seed
    set_global_seed(config["seed"])
    
    if args.mode == 'train':
        train_gnn(config)
    
    elif args.mode == 'inference':
        logger.info("Start Inference")
        infer_gnn(config)
    
    

if __name__ == "__main__":
    main()
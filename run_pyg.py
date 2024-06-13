from loguru import logger


from util import create_parser, set_global_seed, setup_mlflow, setup_logger, init_config, update_config
from pyg.train import train_gnn
from pyg.inference import infer_gnn

def main():
    
    # Initialize configuration
    config = init_config()
    
    # Setup arguments
    parser = create_parser(config)
    args = parser.parse_args()
    
    # Update configurations 
    config = update_config(dict(vars(config)), vars(args))

    # Setup logger
    setup_logger()
    
    
    # Setup MLflow
    setup_mlflow(config)
        
    # Set seed
    set_global_seed(config["general_config"]["seed"])
    
    if args.mode == 'train':
        train_gnn(config)
    
    elif args.mode == 'inference':
        logger.info("Start Inference")
        infer_gnn(config)
    
    

if __name__ == "__main__":
    main()
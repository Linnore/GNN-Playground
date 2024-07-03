from loguru import logger


from utils_dgl import create_parser, set_global_seed, setup_mlflow, setup_logger, init_config, update_config
from dgl_m.train import train_gnn
from dgl_m.inference import infer_gnn
from dgl_m.evaluate import eval_gnn

def main():
    
    # Initialize configuration
    config = init_config()
    
    # Setup arguments
    args = create_parser(config)
    
    # Update configurations 
    config = update_config(dict(vars(config)), vars(args))

    # Setup logger
    setup_logger(args)
    
    
    # Setup MLflow
    setup_mlflow(config)
        
    # Set seed
    set_global_seed(config["general_config"]["seed"])
    
    if args.mode == 'train':
        train_gnn(config)
    
    if args.mode == 'evaluate':
        eval_gnn(config)
    
    elif args.mode == 'inference':
        infer_gnn(config)
    

if __name__ == "__main__":
    main()
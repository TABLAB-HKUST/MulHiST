import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True
    

    # Create directories if not exist.
    if not os.path.exists(config.model_name):
        os.makedirs(config.model_name +'/logs')
        os.makedirs(config.model_name +'/models')
        os.makedirs(config.model_name +'/samples-' + config.model_name)
    if not os.path.exists(config.model_name +'/results-'+ str(config.test_iters)):
        os.makedirs(config.model_name +'/results-'+ str(config.test_iters))
        os.makedirs(config.model_name +'/results-%s/AF'%str(config.test_iters))
        os.makedirs(config.model_name +'/results-%s/HE'%str(config.test_iters))
        os.makedirs(config.model_name +'/results-%s/PAS'%str(config.test_iters))
        os.makedirs(config.model_name +'/results-%s/MT'%str(config.test_iters))


    # Data loader.  
    data_loader = get_loader(config)

    # Solver for training and testing StarGAN.
    solver = Solver(data_loader, config)

    if config.mode == 'train':
        solver.train()
        
    elif config.mode == 'test':
        solver.test()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=4, help='dimension of domain labels, including source-unstained and target-stained domains')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--crop_size', type=int, default=128, help='image resolution')

    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=3, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=5, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_idt', type=float, default=0, help='weight for identity loss')        

    parser.add_argument('--lambda_gp', type=float, default=5, help='weight for gradient penalty')
    parser.add_argument('--netD', type=str, default='m_scale', choices=['basic', 'm_scale'])
    parser.add_argument('--netG', type=str, default='adain', choices=['basic', 'm_scale', 'adain'])
    parser.add_argument('--style_fusion', type=str, default='adain', choices=['basic', 'm_scale', 'adain'])
    # Training configuration.
    parser.add_argument('--dataroot', type=str, required=True,, help='the path of training data')
    parser.add_argument('--test_dataroot', type=str, default='/data/test', help='the path of testing data')
    parser.add_argument('--AF', type=str, default='AF', help='AF image (domain 1)')    
    parser.add_argument('--HE', type=str, default='HE', help='HE image (domain 2)')  
    parser.add_argument('--PAS', type=str, default='PAS', help='PAS image (domain 3)')  
    parser.add_argument('--MT', type=str, default='MT', help='MT image (domain 4)')

    parser.add_argument('--scale_A', type=float, default=1, help='resize slide ratio for unstained WSI')
    parser.add_argument('--scale_B', type=float, default=1, help='resize slide ratio for stained WSI') 
        
    parser.add_argument('--train_data', type=int, default=10000, help='the size of training dataset, no use')
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')

    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=50000, help='number of iterations for decaying lr') #100000
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=1, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    
    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=100000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Directories.
    parser.add_argument('--model_name', type=str, default='mulhist-modelname')
    parser.add_argument('--celeba_image_dir', type=str, default='data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
    parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
    parser.add_argument('--result_dir', type=str, default='results-80k')

    # Step size.
    parser.add_argument('--log_step', type=int, default=500)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--model_save_step', type=int, default=5000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)



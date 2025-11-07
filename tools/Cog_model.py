#python inner libraries 
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import sys
import os
from pathlib import Path
from time import time
import math
from abc import ABC, abstractmethod

# Add parent directory to path to allow imports
parent_dir = str(Path(__file__).parent.parent)
work_dir = str(Path(__file__).parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if work_dir not in sys.path:
    sys.path.append(work_dir)
#external libraries
import torch
import torch.nn.init as init
import numpy as np
import torch.nn as nn
import os
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from typing import Optional, Tuple, Union, Callable,List, Dict, Any
import torch._dynamo
torch._dynamo.config.suppress_errors = True
#my libraries
from losses import Triplets_cross_entropy_loss
from triplets_data_set import OddOneOutDataset,TripletsDataLoader   
from similarity_functions import dot_product
from utils import set_optimizer,calculate_batch_size,consolidate_choice_counts,calculate_NLL
from reg_funcs import RegularizationFunction
from constraints import constraints_wrapper,OrthogonalScaledLinear

class CogModel(nn.Module):
    def __init__(self,
        model_name:str, #the name of the model
        images_features_dim:int, #the dimension of the images_features
        work_with_S_matrix:bool = True, #if we want to work with a precomputed S_matrix
        similarity_function: Callable = dot_product, #the similarity function to be used for the task.
        temp:float = 1.0, #the temperature parameter for the similarity function.
        dim_reduction_method :Optional[str] = None, #the methods to reduce the dimension of the image embeddings.
        constraints:Union[str,Tuple[str,int],None] = None, #the constraints to be applied to the model parameters.
        device = 'cuda' if torch.cuda.is_available() else 'cpu', #the device to be used for the task.
        dtype = torch.float32, #the data type to be used for the task.
        max_memory_gb: float = 12.0, #the maximum memory to be used for the task.
        ):
        """
        CogModel is a class that implements the Model predicting the similarity between images in the odd-one-out task.
        Parameters:
            model_name: str, the name of the model.
            model (optional): VisionModel, the model to who's representations we use to predict the similarity between images.
            similarity_function : Callable, the similarity function to be used for the task. (default: dot_product)
            temp: float, the temperature parameter for the similarity function. (default: 1.0)
            dim_reduction_matrix: torch.Tensor, the matrix to reduce the dimension of the image embeddings. (default: None)
            device: str, the device to be used for the task. (default: 'cuda' if available, else 'cpu')
            dtype: torch.dtype, the data type to be used for the task. (default: torch.float32)
            images_features_dim: int, the dimension of the images_features.
        """
        super(CogModel, self).__init__()
        
        # Register device buffer (empty tensor on desired device)
        self.register_buffer('_device_tensor', torch.tensor([], device=device))
    
        self.model_name = model_name
        #Define the similarity function
        self.similarity_function = similarity_function  # Define the similarity function
        self.dtype = dtype # Define the data type to be used for the task
        self.temp = nn.Parameter(torch.tensor(temp, device=device)) # Define the temperature parameter for the similarity function
        self.temp.requires_grad = False # Freeze the temperature parameter
        #initialize  the W transformation matrix to be the identity matrix
        self.dim_reduction_type = dim_reduction_method
        
        self.constraint = constraints
        if type(constraints) == tuple:
            name,k_dim = constraints
            assert name in ["rectangular","Rectangular"] and type(k_dim) == int and k_dim < images_features_dim, "The constraints must be a tuple of the form (rectangular/Rectangular,k_dim) where name is a string and k_dim is an integer and lower than the images_features_dim"
            self.W_mat = nn.Linear(images_features_dim, k_dim, bias=False, dtype=dtype, device=device)
            # Create a contiguous weight parameter in the required shape:
            self.W_mat.weight = nn.Parameter(torch.eye(k_dim, images_features_dim, dtype=dtype, device=device))
            self.constraint = name
            self.k_dim = k_dim
        elif constraints == 'orthogonal':
            self.W_mat = OrthogonalScaledLinear(images_features_dim,images_features_dim,device=device,dtype=dtype)
        else:
            self.W_mat = nn.Linear(images_features_dim,images_features_dim,bias=False,dtype=dtype,device=device)
            init.eye_(self.W_mat.weight)# Initialize the W transformation matrix to be the identity matrix giving us a zero shot model
            if constraints is not None:
                assert type(constraints) == str, "The constraints must be a string or a tuple (for the rectangular/Rectangular constraint) "
                constraints_wrapper(constraints,self.W_mat)

        self.max_memory_gb = max_memory_gb
        self.images_features_dim = images_features_dim
        self.work_with_S_matrix = work_with_S_matrix
    def apply_W_matrix(self,images_features:torch.Tensor):
        if isinstance(self.W_mat, (nn.Linear, nn.Module)):
            return self.W_mat(images_features)
        else:
            return images_features@self.W_mat
    
    
    @property
    def device(self):
        """Get the model's device from its device buffer"""
        return self._device_tensor.device
    
    def to(self, *args, **kwargs):
        """Override to() to ensure device buffer moves with the model"""
        super().to(*args, **kwargs)
        if hasattr(self, '_device_tensor'):
            device = next(self.parameters()).device
            self._device_tensor = self._device_tensor.to(device)
        return self
    
    def calculate_S_matrix(
        self,
        images_features: Union[torch.Tensor],  # the images data
        W_matrix: Optional[Union[nn.Linear,torch.Tensor]] = None,  # transformation matrix
        similarity_function: Optional[Callable] = None,
        temp: Optional[float] = None,
        *,
        self_instance=None,  # Optional self for standalone mode
    ) -> torch.Tensor:  
        """
        Calculate the similarity matrix between images using the specified similarity function.
        This function can be used as a static method for the CogModel class or as a method for the CogModel instance.
        
        Parameters:
            images_data: The images data to calculate similarity matrix for.
            Can be a ConceptsDataSet, raw tensor of image images_features, or OddOneOutDataset.
            W_mat: Transformation matrix to apply to images_features before calculating similarity. Defaults to the instance's W_mat.
            similarity_function: The similarity function to be used for the task. Defaults to the instance's similarity_function.
            temp: The temperature parameter for scaling similarity. Defaults to the instance's temp.
        
        Returns:
            torch.Tensor: Similarity matrix scaled by the temperature parameter.
        """
        # Allow function to work as standalone if self_instance is provided
        model = self_instance or self
        # Get the images data from the datasets if needed

        # Ensure images_data is a valid tensor
        assert len(images_features.shape) == 2, "images_data must be a 2D tensor"
        if hasattr(self,'k_dim'):
            assert images_features.shape[1] == self.k_dim, "images_data must match the images_features dimension of the model"
        else:
            assert images_features.shape[1] == model.images_features_dim, (
                "images_data must match the images_features dimension of the model"
            )

        # Use instance or provided parameters
        similarity_function = similarity_function or model.similarity_function
        temp = temp or model.temp 

        # Apply transformation and similarity function
        if W_matrix is not None:
            if isinstance(W_matrix, nn.Linear):
                transformed_features = W_matrix(images_features)
            else:
                transformed_features = images_features@W_matrix
            S_mat = similarity_function(transformed_features, transformed_features.T)
        else:
            S_mat = similarity_function(images_features, images_features.T)
            
        return S_mat 
    
    def get_S_pairs_matmul_batch(
        self,
        triplets: torch.Tensor, 
        images_features: Union[torch.Tensor], 
        max_memory_gb: float = 12.0
        ) -> torch.Tensor:
        """
        Calculate similarity scores for triplets using matrix multiplication with batching.
        
        Parameters:
            triplets (torch.Tensor): An Nx3 tensor containing triplet indices (i,j,k)
            images_features (Union[torch.Tensor]): Feature vectors for all items,
                can be raw tensor, OddOneOutDataset, or ConceptsDataSet
            max_memory_gb (float, optional): Maximum GPU memory to use in gigabytes. Defaults to 12.0.
                If None, uses self.max_memory_gb.
        
        Returns:
            torch.Tensor: Nx3 tensor containing similarity scores for each triplet:
                - [:,0]: similarity between j and k 
                - [:,1]: similarity between i and k
                - [:,2]: similarity between i and j
        """
        #get the device
        device = self.device if self.device is not None else images_features.device 
        #get the number of triplets 
        n_triplets = triplets.shape[0]
        #check if the images_features match the images_features dimension of the model
        assert images_features.shape[1] == self.images_features_dim, "images_features must match the images_features dimension of the model"
        #get the images_features dimension
        feature_dim = self.images_features_dim
        #get the maximum memory to use
        max_memory_gb = self.max_memory_gb if max_memory_gb is None else max_memory_gb 
        #calculate the batch size
        batch_size = calculate_batch_size(n_triplets, feature_dim,max_memory_gb)
        #calculate the number of batches
        n_batches = math.ceil(n_triplets / batch_size)
        #initialize the result tensor
        result = torch.empty(n_triplets, 3, device=device)
        
        for i in range(n_batches):
            #empty the cache
            torch.cuda.empty_cache()
            #get the start and end indices of the batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_triplets)
            #get the batch triplets
            batch_triplets = triplets[start_idx:end_idx]
            #get the batch images_features
            batch_images_features = images_features[batch_triplets]
            #calculate the similarity scores
            S_pairs_full = self.similarity_function(batch_images_features, batch_images_features)
            #store the similarity scores in the result tensor
            result[start_idx:end_idx] = torch.stack([
                S_pairs_full[:,1,2],  # x_j·x_k
                S_pairs_full[:,0,2],  # x_i·x_k
                S_pairs_full[:,0,1]   # x_i·x_j
            ], dim=1)
            #delete the batch images_features and similarity scores
            del batch_images_features, S_pairs_full
        return result   
    
    def forward(self,
                triplets:torch.Tensor,# and NX3 tensor of triplets (i,j,k)
                images_features:torch.Tensor, #the images data
                ):
        """
        Forward pass of the COG model.
        Para:
            triplets (torch.Tensor): An Nx3 tensor containing triplet indices (i,j,k)
            images_data (Union[ConceptsDataSet, torch.Tensor, OddOneOutDataset, None]): 
                The input image images_features/embeddings. Can be:
                - ConceptsDataSet: Dataset containing concept images
                - torch.Tensor: Raw feature tensor
                - OddOneOutDataset: Dataset for odd-one-out task
                - None: For using precomputed S_matrix
            work_with_S_matrix (bool, optional): If True, uses a precomputed similarity matrix
                instead of computing similarities on-the-fly. Defaults to False.
                
        Returns:
            torch.Tensor: An Nx3 tensor containing either:
                For each triplet (i,j,k), returns scores for:
                    - [:,0]: similarity between j and k
                    - [:,1]: similarity between i and k  
                    - [:,2]: similarity between i and j
        """
        if len(images_features.shape) >2:
            images_features = self.online_images_feauters(images_features)
        # Ensure images_features reside on the same device as the model
        if images_features.device != self.device:
            images_features = images_features.to(self.device)
        # Ensure images_features use the same dtype as the model
        if images_features.dtype != self.dtype:
            images_features = images_features.to(self.dtype)
        # Apply the W transformation to images_features
        transformed_images_features = self.apply_W_matrix(images_features)
        if self.work_with_S_matrix:
            # Handle the case where we want to work with a precomputed S_matrix
            S_matrix = self.calculate_S_matrix(transformed_images_features)
            i, j, k = triplets.T  # transpose to get (3, N) 
            S_pairs = S_matrix[torch.stack([j,i,i]), torch.stack([k,k,j])].T
        else:
            # a direct method to compute the similarity scores
            S_pairs = self.get_S_pairs_matmul_batch(triplets,transformed_images_features)
        
        #apply the temperature scaling 
        S_pairs = S_pairs/self.temp.clamp(min=1e-7) #clamp to avoid division by zero
        return S_pairs
    
    def online_images_feauters(self,images:torch.Tensor):
        """
        Calculate the images features for the images
        """
        assert self.model is not None, "Set model to extract images features"
        images_features = self.model(images)
        return images_features
    
    def online_images_transformed(self,images:torch.Tensor):
        """
        Calculate the images features for the images
        """ 
        assert self.model is not None, "Set model to extract images features"
        images_features = self.online_images_feauters(images)
        transformed_images_features = self.W_mat(images_features) if isinstance(self.W_mat, nn.Module) else self.W_mat@images_features
        return transformed_images_features
        
    def set_zero_shot_S_matrix(
        self,
        images_features:torch.Tensor
        ):
        self.S_matrix_zero_shot = self.calculate_S_matrix(images_features)
    
    def reset_W_mat(self): 
        if isinstance(self.W_mat, nn.Sequential) or isinstance(self.W_mat, nn.Module):
            # For ShallowDNN, use the built-in identity-like initialization
            self.W_mat.init_identity_like()
        elif hasattr(self.W_mat, 'weight'):
            # For nn.Linear and similar
            init.eye_(self.W_mat.weight)
            #check if self.constraint exist 
            if hasattr(self,'constraint'):  
                if self.constraint is not None:
                    assert type(self.constraint) == str, "The constraints must be a string or a tuple (for the rectangular/Rectangular constraint) "
                    constraints_wrapper(self.constraint,self.W_mat)
        
        #zero any gradients in the W_mat
        for param in self.W_mat.parameters():
            if param.grad is not None:
                param.grad = None
    
    def fit_W_matrix(self,
        train_set:OddOneOutDataset,
        images_features:torch.Tensor,
        optimizer_name:str ='LBFGS',
        reg_func :Optional[str] = "eye_distance",
        loss_func = Triplets_cross_entropy_loss(),
        reg_con = 0.01,
        constraints = None,
        num_epochs = 100,
        verbose = True,
        rel_tol = 1e-7,
        optimizer_kwargs = None,
        ):
        # Ensure all parameters have requires_grad set to True
        for param in self.W_mat.parameters():
            param.requires_grad = True
        # Ensure the temperature parameter is not optimized
        self.temp.requires_grad = False
        optimizer = set_optimizer(optimizer_name,model_parameters=self.parameters())     
        losses_list = []
        self.reg_con = reg_con
        reg_func = RegularizationFunction(reg_func_name=reg_func,lambda_reg=reg_con,device=self.device)
        temp_start = self.temp
        # Store the regularization constant
        
        for epoch in range(num_epochs):
            # Shuffle the train_set each epoch
            # New code using the global RNG:
            torch.manual_seed(epoch)  # Set the seed globally
            indexes = torch.randperm(len(train_set), device=self.device)
            x_train, y_train = train_set[indexes]
            
            def closure():
                optimizer.zero_grad()  # Zero the gradients
                similarity_scores = self.forward(triplets=x_train,images_features=images_features)  # Get the similarity scores
                if reg_con!=0:
                # Calculate regularization penalty based on W_mat type
                    if hasattr(self.W_mat, 'weight'):
                        reg_penalty = reg_func(self.W_mat.weight)  # The penalty for the regularization function
                
                    elif hasattr(self.W_mat, 'shallow'):
                        reg_penalty = reg_con*sum(p.pow(2).sum() for p in self.W_mat.parameters())   #L2 regularization of the weights for all linear layers in the shallow DNN
                else:
                    reg_penalty = 0.0
                loss = loss_func(similarity_scores, y_train) + reg_penalty #calculate the loss
                loss.backward() #backpropagate the loss)
                return loss
            # Call optimizer.step with closure   
            loss = optimizer.step(closure) #step the optimizer
            # assert self.temp == temp_start, "The temperature is being optimized there is a bug in the optimizer"
            losses_list.append(loss.item()) #append the loss to the list
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss}") 
            if epoch > 1:
                max_loss = max(abs(losses_list[epoch-1]),abs(losses_list[epoch])) + 1e-10
                relative_diff = abs(losses_list[epoch-1] - losses_list[epoch])/max_loss  # Calculate the relative difference
                if relative_diff < rel_tol:  # Stop the training when the tolerance criterion is met
                    break
                
        return losses_list
    
    def pred(self,
            triplets:torch.Tensor,
            images_features:torch.Tensor,
            return_image_index:bool = False,
            log_probs:bool = False,
            probs:bool = False,
            ):  
        with torch.no_grad():
            S_pairs = self.forward(triplets=triplets,images_features=images_features)
            if log_probs:
                S_pairs = torch.log_softmax(S_pairs,dim=1)
                if probs:
                    S_pairs = S_pairs.exp()
                return S_pairs
            odd_one_out_position = torch.argmax(S_pairs,dim=1)
            if return_image_index:
                triplets_image_index = triplets.to('cuda')[torch.arange(len(triplets)),odd_one_out_position]
                return triplets_image_index
        return odd_one_out_position
    
    def one_human_pred(self,
                   triplets:torch.Tensor,
                   images_features:torch.Tensor,
                   ):
            # Move the batch and images_features to the device
            triplets = triplets.to(self.device)
            images_features = images_features.to(self.device)
            S_pairs = self.forward(triplets,images_features)
            conditional_log_prob_matrix = F.log_softmax(S_pairs,dim=1) # softmax over triplets 
            # Simulate the human response as a multinomial distribution over triplet choice probabilities
            simulated_choice_counts = torch.distributions.Multinomial(
                                                    logits=conditional_log_prob_matrix,
                                                    total_count = 1,).sample() #  n_triplets x n_choices
            # Convert the one-hot encoded simulated_choice_counts to the choice index
            choice_position = torch.argmax(simulated_choice_counts,dim=1)
            return choice_position
    
    def test_model(self,
                test_set:OddOneOutDataset,
                images_features:torch.Tensor,
                batch_size:int = 'full',
                ):
        if batch_size == "full":
            batch_size = len(test_set)
        # Create the dataloader for evaluation
        test_loader = TripletsDataLoader(test_set, batch_size=batch_size, shuffle=False)
        accuracy_list = []
        NLL_list = []

        for batch_X, batch_y in test_loader:
            # Move the batch to the device
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            preds = self.pred(batch_X,images_features)
            S_pairs = self.forward(batch_X,images_features)
            NLL_list.append(calculate_NLL(S_pairs,batch_y))
            if len(batch_y.shape) == 1: # for the case y is represent the index of the correct choice
                accuracy = (preds == batch_y).float().mean()
            else:
                # get model predictions in y at each triplet
                model_preds_in_y = torch.gather(batch_y, 1, preds.unsqueeze(1)).squeeze()
                #sum the one hot encoded y at each triplet
                batch_y_summed = batch_y.sum(dim=1)
                #calculate the accuracy as the mean of the model predictions in y divided by the sum of the one hot encoded y
                accuracy = (model_preds_in_y/batch_y_summed).mean()
            #append the accuracy to the list
            accuracy_list.append(accuracy)
        #calculate the mean accuracy
        mean_accuracy = torch.tensor(accuracy_list).mean()
        mean_NLL = torch.tensor(NLL_list).mean()
        return mean_accuracy, mean_NLL

    def reset_temp(self):
        self.temp.data = torch.tensor(1.0)
        self.temp.requires_grad = False
        
        
    def estimate_noise_ceiling(self,
                                S_pairs:torch.Tensor,
                                ):
        """
        Calculate the trial-level noise ceiling according to a model conditional_log_prob_matrix
        args:
        S_triplet: torch.tensor, the similarity scores for the triplet
        returns:
        p (scalar torch.tensor) the probability of correctly predicting a trial
        entropy (scalar torch.tensor) the entropy of the model's prediction
        """
        conditional_log_prob_matrix = F.log_softmax(S_pairs,dim=1) # softmax over triplets 
        max_log_p, _ = conditional_log_prob_matrix.max(dim=1)
        max_p = max_log_p.exp()
        entropy = F.cross_entropy(S_pairs,conditional_log_prob_matrix,reduction='mean') #if cross entropy is P(x)log(Q(x)) when P=Q its just entropy
        expected_accuracy = max_p.mean(dim=0)
        return expected_accuracy, entropy
    
    
    
    def fit_model_temp(self,
        data_set:OddOneOutDataset,
        images_features:torch.Tensor,
        optimizer_name:str ='LBFGS',
        loss_func = Triplets_cross_entropy_loss(),
        num_epochs = 100,
        batch_size = 10,
        verbose = True,
        fit_to_noise_ceiling:bool = True,
        target_noise_ceiling:float = 0.67,
        rel_tol:float = 1e-9
        ):
        # Disable gradient calculation for the W matrix
        for param in self.W_mat.parameters():
            param.requires_grad = False
        
        # Verify that W matrix parameters are frozen
        w_params_frozen = all(not param.requires_grad for param in self.W_mat.parameters())
        assert w_params_frozen, "The W matrix is still being optimized"
        # Enable gradient calculation for the temp
        self.temp.requires_grad = True

        
        # Set up optimizer for temperature parameter
        optimizer = set_optimizer(optimizer_name, self.parameters(),**{'tolerance_change': 1e-7})
        
        # Create data loader
        loader = TripletsDataLoader(data_set, batch_size=batch_size, shuffle=True)
        losses_list = []
        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch_X, batch_y in loader:
                # Move batch to device
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)            

                
                def temp_closure(): 
                    optimizer.zero_grad() #zero the gradients
                    # Get similarity scores
                    S_pairs = self.forward(batch_X, images_features)
                    
                    # Calculate loss
                    if fit_to_noise_ceiling: #fit to the noise ceiling
                        expected_accuracy,_ = self.estimate_noise_ceiling(S_pairs) #calculate the model "noise ceiling
                        loss = (expected_accuracy - target_noise_ceiling)**2
                    else:
                        loss = loss_func(S_pairs, batch_y)    
                    #backpropagate the loss
                    loss.backward()
                    return loss
                # Call optimizer.step with closure   
                loss = optimizer.step(temp_closure)
                epoch_loss += loss.item()
            epoch_loss /= len(loader)
            losses_list.append(epoch_loss)
            if verbose:
                print(f"epoch: {epoch}/{num_epochs}, temp: {self.temp.item()}, loss: {epoch_loss}")
            # Check the relative tolerance stop condition
            if epoch > 1:
                max_loss = max(abs(losses_list[epoch-1]),abs(losses_list[epoch])) + 1e-10
                relative_diff = abs(losses_list[epoch-1] - losses_list[epoch])/max_loss  # Calculate the relative difference
                if relative_diff < rel_tol:  # Stop training when the relative difference is below the tolerance
                    break
        
        # Disable gradient calculation for temp
        self.temp.requires_grad = False
    
    def simulate_human_response(self,
                                triplets:Union[torch.Tensor,OddOneOutDataset],
                                images_features:torch.Tensor,
                                N_participants:int = 100,
                                batch_size:Union[int,str] = 100,
                                seed: Optional[int] = None):
        unique_triplets = []
        choice_counts = []
        # Create the dataloader used for simulation
        if type(batch_size) == str:
            if batch_size == "full":
                batch_size = len(triplets)
            else:
                raise ValueError(f"batch_size must be an integer or 'full', not {batch_size}")
            
        data_loader = TripletsDataLoader(triplets, batch_size=batch_size, shuffle=True,only_triplets=True)
        # Iterate over the batches
        for batch_triplets in data_loader:
            # Move the batch and images_features to the device
            batch_triplets = batch_triplets.to(self.device)
            images_features = images_features.to(self.device)
            S_pairs = self.forward(batch_triplets,images_features)
            conditional_log_prob_matrix = F.log_softmax(S_pairs,dim=1) # softmax over triplets 
            # Simulate the human response as a multinomial distribution over triplet choice probabilities
            simulated_choice_counts = torch.distributions.Multinomial(
                                                    logits=conditional_log_prob_matrix,
                                                    total_count = 1,).sample() #  n_triplets x n_choices
            # Consolidate the choice counts so each row represents a unique triplet
            unique_triplets_batch, consolidated_choice_counts_batch = consolidate_choice_counts(batch_triplets, simulated_choice_counts)
            unique_triplets.append(unique_triplets_batch)
            choice_counts.append(consolidated_choice_counts_batch)  
        # Concatenate the unique triplets and choice counts
        unique_triplets = torch.cat(unique_triplets,dim=0)
        consolidated_choice_counts = torch.cat(choice_counts,dim=0)
        # Shuffle the triplets and choice counts
        shuffled_indexes = torch.randperm(len(consolidated_choice_counts), device = self.device)
        unique_triplets = unique_triplets[shuffled_indexes]
        consolidated_choice_counts = consolidated_choice_counts[shuffled_indexes]
        if N_participants == 1:  # Convert the choice counts to the choice position index
            choice_position = torch.argmax(consolidated_choice_counts,dim=1)
            return unique_triplets, choice_position
        else:
            return unique_triplets, consolidated_choice_counts
        
    def save(self, save_folder_path: str, file_name: Optional[str] = None):
        """
        Save model to disk using model_io functionality.
        
        Args:
            save_folder_path: Directory path where to save the model
            file_name: Optional name for the saved file (default: model_name.pt)
        """
        from model_io import save_cog_model
        return save_cog_model(self, save_folder_path, file_name)

    @classmethod
    def load(cls, save_folder_path: str, file_name: Optional[str] = None, device=None, dtype=None) -> 'CogModel':
        """
        Load model from disk using model_io functionality.
        
        Args:
            save_folder_path: Directory path where model is saved
            file_name: Optional name of the saved file (default: looks for .pt file)
            device: Target device (default: CUDA if available, else CPU)
            dtype: Target dtype (default: saved dtype)
            
        Returns:
            Loaded CogModel instance
        """
        from model_io import load_cog_model
        return load_cog_model(save_folder_path, file_name, device=device, dtype=dtype)
    

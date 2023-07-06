import torch
from pytorch_pretrained_bert import BertModel
from torch import nn
from torch.nn import functional as F
import xxhash


class HashTensorWrapper2():
    def __init__(self, tensor):
        self.tensor = tensor
        self.hashcrap = torch.arange(self.tensor.numel(), device=self.tensor.device).reshape(self.tensor.size())

    def __hash__(self):
        if self.hashcrap.size() != self.tensor.size():
            self.hashcrap = torch.arange(self.tensor.numel(), device=self.tensor.device).reshape(self.tensor.size())
        return hash(torch.sum(self.tensor*self.hashcrap))

    def __eq__(self, other):
        return torch.all(self.tensor == other.tensor)
    
### https://discuss.pytorch.org/t/how-to-put-tensors-in-a-set/123836/7
class HashTensorWrapper():
    def __init__(self, tensor):
        self.tensor = tensor
        self.hash = xxhash.xxh64()
        self.hash.update(tensor.cpu().numpy().tobytes())

    def __hash__(self):
        #return hash(self.tensor.cpu().numpy().tobytes())
        return self.hash.intdigest()

    def __eq__(self, other):
        return torch.all(self.tensor == other.tensor)

class BertMultiClassifier(nn.Module):
    def __init__(self, bert_model_path, labels_count, hidden_dim=768, dropout=0.1,
                 enable_cache=False, cache_embedding={}, cache_masked_tokens={}, random_state=1):
        super().__init__()

        self.config = {
            'bert_model_path': bert_model_path,
            'labels_count': labels_count,
            'hidden_dim': hidden_dim,
            'dropout': dropout,
        }
        if labels_count == 2:
            labels_count = 1 ## force label_count to 1 to use sigmoid

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_cache = enable_cache
        self.cache_embedding = cache_embedding
        self.cache_masked_tokens = cache_masked_tokens
        self.random_state = random_state

        self.bert = BertModel.from_pretrained(bert_model_path)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, labels_count)

        if labels_count == 1:
            self.output_layer = nn.Sigmoid()
        else:
            self.output_layer = nn.Softmax()
        #self.sigmoid = nn.Sigmoid()
        self.pooler = self.bert.pooler ### we get the pooler from BertModel to use it in the MLP


    def forward(self, tokens, masks):
        if self.training and self.enable_cache:
            masked_tokens = (tokens * masks)
            masked_tokens_stack = []
            encoded_layers_stack = []
            for t in range(masked_tokens.shape[0]):  ## has each tensor representing a sentence
                new_masked_tokens = masked_tokens[t,:] ## get the masked tokens for each sentence
                new_masked_tokens_wrapper = HashTensorWrapper(new_masked_tokens) ## wrap the masked tokens in a HashTensorWrapper
                new_masked_tokens_hash = hash(new_masked_tokens_wrapper) ## hash the masked tokens

                ## look for the hash of the masked tokens in the cache
                if new_masked_tokens_hash in self.cache_embedding.keys():  ## if we found the masked tokens in the cache we use the encoded_layer form cache
                    #### Check if the tokes are exaclty the same previously found
                    assert (self.cache_masked_tokens[new_masked_tokens_hash] == new_masked_tokens.detach().clone().cpu().numpy()).all()                   
                    #### Recover the encoded_layers_single_element from the cache
                    encoded_layers_np = self.cache_embedding[new_masked_tokens_hash]
                    #### Load the encoded_layers_single_element from the cache in to the device
                    encoded_layers_single_element = torch.from_numpy(encoded_layers_np).to(self.device)
                else: ## if we didn't find the masked tokens in the cache we use the encoded_layer form bert  
                    ## See: https://vandurajan91.medium.com/random-seeds-and-reproducible-results-in-pytorch-211620301eba
                    torch.manual_seed(self.random_state)             
                    encoded_layers_single_element, _ = self.bert(tokens[t,:].reshape(1,-1), attention_mask=masks[t,:].reshape(1,-1), output_all_encoded_layers=False)
                    
                    ### add the encoded_layers_single_element to the cache
                    self.cache_embedding[new_masked_tokens_hash] = encoded_layers_single_element.detach().clone().cpu().numpy()

                    ### add the masked_tokens to the old_masked_token dictionary
                    self.cache_masked_tokens[new_masked_tokens_hash] = new_masked_tokens.detach().clone().cpu().numpy()
                ## add encoded_layers_single_element to the stack
                encoded_layers_stack.append(encoded_layers_single_element)
                ## add masked_tokens to the stack
                masked_tokens_stack.append(new_masked_tokens)
            
            ## build encoded_layers_stack and masked_tokens_stack from the stack
            encoded_layers = torch.stack(encoded_layers_stack, dim=1)[0] ### rebuild the encoded_layers tensor from the stack and drop first dimension of the stack to match expected tensor dimensions
            rebuild_masked_tokens = torch.stack(masked_tokens_stack)   ### rebuild the masked_tokens tensor from the stack
            assert torch.eq(masked_tokens, rebuild_masked_tokens).all().item() is True

        else:
            #torch.manual_seed(1)
            encoded_layers, _ = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)


        pooled_output = self.pooler(encoded_layers)
        #_, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)
        dropout_output = self.dropout(pooled_output)

        linear_output = self.linear(dropout_output)
        
        proba = self.output_layer(linear_output)

        return proba

class ExtraBertMultiClassifier(nn.Module):
    def __init__(self, bert_model_path, labels_count, hidden_dim=768, mlp_dim=100, 
                 extras_dim=6, dropout=0.1, enable_cache=False, cache_embedding={}, cache_masked_tokens={}, random_state=1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_cache = enable_cache
        self.cache_embedding = cache_embedding
        self.cache_masked_tokens = cache_masked_tokens
        self.random_state = random_state
        self.config = {
            'bert_model_path': bert_model_path,
            'labels_count': labels_count,
            'hidden_dim': hidden_dim,
            'mlp_dim': mlp_dim,
            'extras_dim': extras_dim,
            'dropout': dropout,
        }
        if labels_count == 2:
            labels_count = 1 ## force label_count to 1 to use sigmoid
        self.bert = BertModel.from_pretrained(bert_model_path)
        self.pooler = self.bert.pooler ### we get the pooler from BertModel to use it in the MLP
        #self.dropout = nn.Dropout(p=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + extras_dim, mlp_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            # nn.Linear(mlp_dim, mlp_dim),
            # nn.Dropout(p=dropout),
            # nn.ReLU(),            
            nn.Linear(mlp_dim, labels_count)
        )
        if labels_count == 1:
            self.output_layer = nn.Sigmoid()
        else:
            self.output_layer = nn.Softmax()

    def forward(self, tokens, masks, extras = None):

        if self.training and self.enable_cache:
            masked_tokens = (tokens * masks)
            masked_tokens_stack = []
            encoded_layers_stack = []
            for t in range(masked_tokens.shape[0]):  ## has each tensor representing a sentence
                new_masked_tokens = masked_tokens[t,:] ## get the masked tokens for each sentence
                new_masked_tokens_wrapper = HashTensorWrapper(new_masked_tokens) ## wrap the masked tokens in a HashTensorWrapper
                new_masked_tokens_hash = hash(new_masked_tokens_wrapper) ## hash the masked tokens

                ## look for the hash of the masked tokens in the cache
                if new_masked_tokens_hash in self.cache_embedding.keys():  ## if we found the masked tokens in the cache we use the encoded_layer form cache
                    #### Check if the tokes are exaclty the same previously found
                    assert (self.cache_masked_tokens[new_masked_tokens_hash] == new_masked_tokens.detach().clone().cpu().numpy()).all()                   
                    #### Recover the encoded_layers_single_element from the cache
                    encoded_layers_np = self.cache_embedding[new_masked_tokens_hash]
                    #### Load the encoded_layers_single_element from the cache in to the device
                    encoded_layers_single_element = torch.from_numpy(encoded_layers_np).to(self.device)
                else: ## if we didn't find the masked tokens in the cache we use the encoded_layer form bert  
                    ## See: https://vandurajan91.medium.com/random-seeds-and-reproducible-results-in-pytorch-211620301eba
                    torch.manual_seed(self.random_state)             
                    encoded_layers_single_element, _ = self.bert(tokens[t,:].reshape(1,-1), attention_mask=masks[t,:].reshape(1,-1), output_all_encoded_layers=False)
                    
                    ### add the encoded_layers_single_element to the cache
                    self.cache_embedding[new_masked_tokens_hash] = encoded_layers_single_element.detach().clone().cpu().numpy()

                    ### add the masked_tokens to the old_masked_token dictionary
                    self.cache_masked_tokens[new_masked_tokens_hash] = new_masked_tokens.detach().clone().cpu().numpy()
                ## add encoded_layers_single_element to the stack
                encoded_layers_stack.append(encoded_layers_single_element)
                ## add masked_tokens to the stack
                masked_tokens_stack.append(new_masked_tokens)
            
            ## build encoded_layers_stack and masked_tokens_stack from the stack
            encoded_layers = torch.stack(encoded_layers_stack, dim=1)[0] ### rebuild the encoded_layers tensor from the stack and drop first dimension of the stack to match expected tensor dimensions
            rebuild_masked_tokens = torch.stack(masked_tokens_stack)   ### rebuild the masked_tokens tensor from the stack
            assert torch.eq(masked_tokens, rebuild_masked_tokens).all().item() is True

        else:
            #torch.manual_seed(1)
            encoded_layers, _ = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)
            
        pooled_output = self.pooler(encoded_layers)
        ## check if pooled_output is the same as pooled_output_internal
        # match = torch.all(pooled_output.eq(pooled_output_internal))
        # if match:
        #     print("pooled_output and pooled_output_internal match")
        # else:
        #     print("pooled_output and pooled_output_internal don't match")

        dropout_output = self.dropout(pooled_output)
        dropout_output = (dropout_output + 1) / 2 ## normalize to 0-1
        if extras is not None:
            concat_output = torch.cat((dropout_output, extras), dim=1) ## if we have extra features we concat them to the pooled_output
        else:
            concat_output = dropout_output ## if we don't have extra features we use the pooled_output
        mlp_output = self.mlp(concat_output)
        # proba = self.sigmoid(mlp_output)
        proba = self.output_layer(mlp_output)

        return proba

class LinearMultiClassifier(nn.Module):
    def __init__(self, labels_count, extras_dim=6, dropout=0.1):
        super().__init__()

        self.config = {
            'labels_count': labels_count,
            'extras_dim': extras_dim,
        }

        if labels_count == 2:
            labels_count = 1 ## force label_count to 1 to use sigmoid

        self.linear = nn.Linear(extras_dim, labels_count)
        #self.softmax = nn.Softmax()
        # self.sigmoid = nn.Sigmoid()

        if labels_count == 1:
            self.output_layer = nn.Sigmoid()
        else:
            self.output_layer = nn.Softmax()

    def forward(self, extras):
        lin_output = self.linear(extras)
        # proba = self.sigmoid(mlp_output)
        proba = self.output_layer(lin_output)

        return proba

class ExtraMultiClassifier(nn.Module):
    def __init__(self, labels_count, mlp_dim=100, extras_dim=6, dropout=0.1):
        super().__init__()

        self.config = {
            'labels_count': labels_count,
            'mlp_dim': mlp_dim,
            'extras_dim': extras_dim,
            'dropout': dropout,
        }
        if labels_count == 2:
            labels_count = 1 ## force label_count to 1 to use sigmoid
        self.mlp = nn.Sequential(
            nn.Linear(extras_dim, mlp_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            # nn.Linear(mlp_dim, mlp_dim),
            # nn.Dropout(p=dropout),
            # nn.ReLU(),            
            nn.Linear(mlp_dim, labels_count)
        )

        if labels_count == 1:
            self.output_layer = nn.Sigmoid()
        else:
            self.output_layer = nn.Softmax()

    def forward(self, extras):

        mlp_output = self.mlp(extras)
        # proba = self.sigmoid(mlp_output)
        proba = self.output_layer(mlp_output)
        
        return proba
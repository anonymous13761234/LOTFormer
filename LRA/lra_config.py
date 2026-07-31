
listops = {
              "dataset":{
                  "train":96000,
                  "dev":2000,
                  "test":2000,
              },
              "model":{
                  "learn_pos_emb":True,
                  "tied_weights":False,
                  "embedding_dim":64, 
                  "transformer_dim":64, 
                  "transformer_hidden_dim":128, 
                  "head_dim":32, 
                  "num_head":2, 
                  "num_layers":2,
                  "vocab_size":32,
                  "max_seq_len":2000,
                  "dropout_prob":0.1,
                  "attention_dropout":0.1,
                  "pooling_mode":"MEAN",
                  "num_classes":10,
              },
                # -------- listops --------
                "training": {
                    "batch_size": 32,
                    "learning_rate": 0.0001,
                    "warmup": 5000,
                    "lr_decay": "cosine",
                    "min_lr_ratio": 0.4,
                    "start_factor": 0.10,
                    "weight_decay": 0,
                    "eval_frequency": 1000,
                    "num_train_steps": 50000,
                    "num_init_steps": 5000,
                    "num_eval_steps": 62,
                    "patience": 10,
                
                },

              "extra_attn_config":{
                  "softmax":{"bz_rate":1,},
                  "softmaxRBF32":{"bz_rate":1},
                  "kernelized":{"bz_rate":1},

                  "sketchedRBF32128":{"bz_rate":1, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
                  "skyformer":{"bz_rate":1, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},

                  "nystrom":{"bz_rate":1,"num_landmarks":128},
                  "linformer":{"bz_rate":1,"linformer_k":128},
                  "performer":{"bz_rate":1,"rp_dim":128, "kernel_type":"exp"}, 
                  "informer":{"bz_rate":1,"in_nb_features":128},
                  "reformer":{"bz_rate":1,"num_hash":2},
                  "bigbird":{"bz_rate":1,"num_random_blocks":3, "block_size":64},
                  "sinkhorn_linear": {
                                        "bz_rate": 1,
                                        "num_refs": 4,          # number of landmarks per head
                                        "sink_eps": 0.1,        # entropic regularization ε
                                        "max_iter": 50,         # Sinkhorn iterations
                                        "attention_eps": 1e-6,   # epsilon for row normalization
                                        "learn_z": True,        # uniform landmark priors by default
                                    },

                  }
          }
pathfinder = {
           "model":{
               "learn_pos_emb":True,
               "tied_weights":False,
               "embedding_dim":64, 
               "transformer_dim":64, 
               "transformer_hidden_dim":128, 
               "head_dim":32,
               "num_head":2, 
               "num_layers":2,
               "vocab_size":512,
               "max_seq_len":1024,
               "dropout_prob":0.1,
               "attention_dropout":0.1,
               "pooling_mode":"MEAN",
               "num_classes": 2,
           },
                # -------- pathfinder --------
            "training": {
                    "batch_size": 128,
                    "learning_rate": 0.0001,
                    "warmup": 5000,
                    "lr_decay": "cosine",
                    "min_lr_ratio": 0.4,
                    "start_factor": 0.10,
                    "weight_decay": 0,
                    "eval_frequency": 1000,
                    "num_train_steps": 60000,
                    "num_init_steps": 5000,
                    "num_eval_steps": 156,
                    "patience": 10,
                
                },

           "extra_attn_config":{
               "softmax":{"bz_rate":1,},
               "softmaxRBF32":{"bz_rate":1},
               "kernelized":{"bz_rate":1},

               "sketchedRBF32128":{"bz_rate":1, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
               "skyformer":{"bz_rate":1, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},

               "nystrom":{"bz_rate":1,"num_landmarks":128},
               "linformer":{"bz_rate":1,"linformer_k":128},
               "performer":{"bz_rate":1,"rp_dim":128, "kernel_type":"exp"}, #rp_dim = nb_features
               "informer":{"bz_rate":2,"in_nb_features":128},
               "bigbird":{"bz_rate":1,"num_random_blocks":3, "block_size":64},
               "reformer":{"bz_rate":1,"num_hash":2},
               "sinkhorn_linear": {
                    "bz_rate": 1,
                    "num_refs": 64,          # number of landmarks per head
                    "sink_eps": 0.1,        # entropic regularization ε
                    "max_iter": 50,         # Sinkhorn iterations
                    "attention_eps": 1e-6,   # epsilon for row normalization
                    "learn_z": True,        # uniform landmark priors by default
                },

           }
       }
retrieval={
              "dataset":{
                  "train":147086,
                  "dev":18090,
                  "test":17437,
              },
              "model":{
                  "learn_pos_emb":True,
                  "tied_weights":False,
                  "embedding_dim":64, 
                  "transformer_dim":64, 
                  "transformer_hidden_dim":128, 
                  "head_dim":32, 
                  "num_head":2, 
                  "num_layers":2,
                  "vocab_size":512,
                  "max_seq_len":4000,
                  "dropout_prob":0.1,
                  "attention_dropout":0.1,
                  "pooling_mode":"MEAN",
                  "num_classes": 2,
              },
            # -------- retrieval --------
            "training": {
                "batch_size": 16,
                "learning_rate": 0.0001,
                "warmup": 5000,
                "lr_decay": "cosine",
                "min_lr_ratio": 0.4,
                "start_factor": 0.10,
                "weight_decay": 0,
                "eval_frequency": 1000,
                "num_train_steps": 50000,
                "num_init_steps": 5000,
                "num_eval_steps": 156,
                "patience": 10,
            
            },

              "extra_attn_config":{
                  "softmax":{"bz_rate":1,},
                  "softmaxRBF32":{"bz_rate":2},
                  "kernelized":{"bz_rate":2},

                  "sketchedRBF32128":{"bz_rate":1, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
                  "skyformer":{"bz_rate":1, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
                  

                  "nystrom":{"bz_rate":1,"num_landmarks":128},
                  "linformer":{"bz_rate":1,"linformer_k":128},
                  "performer":{"bz_rate":1,"rp_dim":128, "kernel_type":"exp"}, #rp_dim = nb_features
                  "informer":{"bz_rate":1,"in_nb_features":128},
                  "bigbird":{"bz_rate":1,"num_random_blocks":3, "block_size":64},
                  "reformer":{"bz_rate":1,"num_hash":2},
                  "sinkhorn_linear": {
                    "bz_rate": 1,
                    "num_refs": 64,          # number of landmarks per head
                    "sink_eps": 0.1,        # entropic regularization ε
                    "max_iter": 50,         # Sinkhorn iterations
                    "attention_eps": 1e-6,   # epsilon for row normalization
                    "learn_z": True,        # uniform landmark priors by default
                },
              }
          }
text={
         "dataset":{
             "train":25000,
             "dev":25000,
             "test":25000,
         },
         "model":{
             "learn_pos_emb":True,
             "tied_weights":False,
             "embedding_dim":64, 
             "transformer_dim":64, 
             "transformer_hidden_dim":128, 
             "head_dim":32, 
             "num_head":2, 
             "num_layers":2,
             "vocab_size":512,
             "max_seq_len":4000, 
             "dropout_prob":0.1,
             "attention_dropout":0.1,
             "pooling_mode":"MEAN",
             "num_classes": 2,
         },
            # -------- text --------
            "training": {
                "batch_size": 32,
                "warmup": 4000,
                "lr_decay": "cosine",
                "learning_rate": 0.0001,
                "min_lr_ratio": 0.4,
                "start_factor": 0.10,
                "weight_decay": 0.0,
                "eval_frequency": 1000,
                "num_train_steps": 50000,
                "num_init_steps": 3000,
                "num_eval_steps": 200,
                "patience": 10,
            
            },

         "extra_attn_config":{
             "softmax":{"bz_rate":1},
             "softmaxRBF32":{"bz_rate":2},
             "kernelized":{"bz_rate":2},

             "sketchedRBF32128":{"bz_rate":1, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
             "skyformer":{"bz_rate":1, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
             
             "nystrom":{"bz_rate":1,"num_landmarks":128},
             "linformer":{"bz_rate":1,"linformer_k":128},
             "performer":{"bz_rate":1,"rp_dim":128, "kernel_type":"exp"}, #rp_dim = nb_features
             "informer":{"bz_rate":1,"in_nb_features":128},
             "bigbird":{"bz_rate":1,"num_random_blocks":3, "block_size":64},
             "reformer":{"bz_rate":1,"num_hash":2},
             "sinkhorn_linear": {
                    "bz_rate": 1,
                    "num_refs": 32,          # number of landmarks per head
                    "sink_eps": 0.1,        # entropic regularization ε
                    "max_iter": 50,         # Sinkhorn iterations
                    "attention_eps": 1e-6,   # epsilon for row normalization
                    "learn_z": True,        # uniform landmark priors by default
                },
         }
     }


image={
        "dataset":{
            "train":45000,
            "dev":5000,
            "test":10000,
        },
        "model":{
            "learn_pos_emb":True,
            "tied_weights":False,
            "embedding_dim":64,
            "transformer_dim":64,
            "transformer_hidden_dim":128,
            "head_dim":32,
            "num_head":2,
            "num_layers":2,
            "vocab_size":256, 
            "max_seq_len":1024,
            "dropout_prob":0.1, 
            "attention_dropout":0.1,
            "pooling_mode":"MEAN",
            "num_classes": 10,
        },
        # -------- image --------
        "training": {
            "batch_size": 64,
            "learning_rate": 0.001,
            "warmup": 1000,
            "lr_decay": "cosine",
            "min_lr_ratio": 0.4,
            "start_factor": 0.10,
            "weight_decay": 0,
            "eval_frequency": 50,
            "num_train_steps": 50000,
            "num_init_steps": 0,
            "num_eval_steps": 20,
            "patience": 10,
        
        },

        "extra_attn_config":{
            "softmax":{"bz_rate":1},

            "softmaxRBF32":{"bz_rate":1}, # for stability
            "kernelized":{"bz_rate":1}, # for stability

            "sketchedRBF32128":{"bz_rate":1, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
            "skyformer":{"bz_rate":1, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},


            "nystrom":{"bz_rate":1,"num_landmarks":128},
            "linformer":{"bz_rate":1,"linformer_k":128},
            "performer":{"bz_rate":1,"rp_dim":128, "kernel_type":"exp"}, #rp_dim = nb_features
            "informer":{"bz_rate":2,"in_nb_features":128},
            "bigbird":{"bz_rate":1,"num_random_blocks":3, "block_size":64},
            "reformer":{"bz_rate":1,"num_hash":2},
            "sinkhorn_linear": {
                    "bz_rate": 1,
                    "num_refs": 8,          # number of landmarks per head
                    "sink_eps": 25,        # entropic regularization ε
                    "max_iter": 5,         # Sinkhorn iterations
                    "attention_eps": 1e-6,   # epsilon for row normalization
                    "learn_z": False,        # uniform landmark priors by default
                },
        }
}

Config = {
    "lra-listops":listops,
    "lra-pathfinder":pathfinder,
    "lra-retrieval":retrieval,
    "lra-text":text,
    "lra-image":image,
}

Config["lra-pathfinder32-curv_contour_length_14"] = Config["lra-pathfinder"]

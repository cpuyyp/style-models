# style models

## important notebooks
- multitask_style_learning_utils.ipynb: All definitions and functions are defined here. To load this from another notebook, simply run ```%run ./multitask_style_learning_utils.ipynb```

To train a multitask model, a simple demo here
```python
# select tasks from 22 datasets
selected_tasks = ['PASTEL_country', 
                  'SARC', 
                  'ShortHumor', 
                 ] 

# put all args into this training_args
training_args = TrainingArgs(selected_tasks=selected_tasks,
                             # select from 'bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', etc.
                             base_model_name='bert-base-uncased',
                             # True: freeze all. Or input an int to freeze first n attention layers.
                             freeze_bert=False, 
                             # use pooler output or the CLS token embedding
                             use_pooler=True, 
                             num_epoch=5,
                             # drop rows that exceed this limit for each dataset 
                             data_limit=30000, 
                            ) # there are some other args, check TrainingArgs definition

# training and evaluating
model = init_model(training_args) # initial model and tokenizer
freeze_model(model, training_args.freeze_bert) # freeze model layers
df_evaluation, df_loss_per_step, model = train_model(model, training_args) # train

# run bertology
eval_dataloader = MultiTaskTestDataLoader(training_args, split='dev') 
attn_entropy, head_importance, preds, labels = compute_heads_importance(model, eval_dataloader, training_args)

# visualize bertology result
imshow(attn_entropy)
imshow(head_importance)
```
- other notebooks named by style1+style2+...: These notebooks contains the scripts and output of training one or more specific model(s).

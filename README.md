# UNLP-2025-Shared-Task-Technique-Classification

## 3rd Place Solution for the UNLP 2025 Shared Task Technique Classification Competition

I'm thrilled to share the approach that led to a 3rd place finish in the UNLP 2025 Shared Task Technique Classification competition, achieving a final Macro F1 score of 0.45519 on the private leaderboard. This write-up details the model, techniques, and experiments employed.

**Overview of the Approach**

The core of the solution involved fine-tuning a large pre-trained transformer model, specifically **xlm-roberta-large**, chosen for its strong multilingual capabilities and large capacity.

- **Model:** xlm-roberta-large from Hugging Face Transformers.
    
- **Preprocessing:** Basic text cleaning was applied, primarily replacing URLs with a [URL] token and normalizing whitespace. Texts were tokenized and padded/truncated to a max_length of 512.
    
- **Feature Engineering:** Minimal explicit feature engineering was performed. Instead, the focus was on leveraging the powerful representations learned by the xlm-roberta-large model. A simple data augmentation technique (random word deletion) was applied during training to improve robustness.
    
- **Validation Strategy:** An 80/20 stratified train-validation split was used, stratifying based on whether a sample contained any manipulation technique (manipulative flag derived from labels) to ensure similar distributions in both sets. The primary evaluation metric was Macro F1-score across all target techniques. Early stopping was implemented based on the validation Macro F1 score (patience=4).
    

**Details of the Submission**

Several key components and techniques contributed to the final performance:

1. **Model Architecture (Custom Head):**
    
    - Instead of directly using AutoModelForSequenceClassification, a custom ManipulationClassifier module was built on top of the base xlm-roberta-large model.
        
    - This involved taking the CLS token output, passing it through a Linear layer (pre_classifier) followed by a GELU activation.
        
    - **Monte Carlo Dropout (MCDropout)-like Averaging:** A crucial element was using multiple dropout layers (nn.ModuleList with 5 instances of nn.Dropout) applied before the final classification layer. The outputs from these different dropout masks were averaged to produce the final logits. This acts as a form of ensembling during inference and can improve robustness and calibration. The dropout rate was set to 0.3.
        
    
    ```
    class ManipulationClassifier(nn.Module):
        def __init__(self, model_name, num_labels):
            super(ManipulationClassifier, self).__init__()
            self.model = AutoModel.from_pretrained(model_name)
    
            # Multiple dropout layers for averaging
            self.dropouts = nn.ModuleList([
                nn.Dropout(CONFIG["dropout_rate"]) for _ in range(5)
            ])
    
            hidden_size = self.model.config.hidden_size
            self.pre_classifier = nn.Linear(hidden_size, hidden_size)
            self.activation = nn.GELU()
            self.classifier = nn.Linear(hidden_size, num_labels)
    
            # Initialization
            nn.init.xavier_normal_(self.pre_classifier.weight)
            nn.init.xavier_normal_(self.classifier.weight)
    
        def forward(self, input_ids, attention_mask):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            sequence_output = outputs[0]
            cls_output = sequence_output[:, 0, :] # Use CLS token
    
            cls_output = self.pre_classifier(cls_output)
            cls_output = self.activation(cls_output)
    
            # Average predictions from multiple dropout masks
            logits = torch.zeros(cls_output.size(0), len(TECHNIQUES)).to(cls_output.device)
            for dropout in self.dropouts:
                logits += self.classifier(dropout(cls_output))
            logits = logits / len(self.dropouts)
    
            return logits
    ```
    
    
2. **Training Configuration:**
    
    - **Optimizer:** AdamW with a learning rate of 1.8e-5 and weight decay of 0.01. Layer-wise weight decay was used (excluding bias and LayerNorm parameters from decay).
        
    - **Scheduler:** A cosine annealing learning rate scheduler with warmup (get_cosine_schedule_with_warmup) was employed, with a warmup ratio of 0.1.
        
    - **Batching:** A physical batch size of 8 was used, combined with gradient_accumulation_steps=4, resulting in an effective batch size of 32.
        
    - **Regularization:** Gradient clipping (1.0) was used to prevent exploding gradients.
        
3. **Handling Class Imbalance and Loss Function:**
    
    - The dataset exhibited significant class imbalance across the different manipulation techniques (verified via analyze_dataset).
        
    - To address this, **Weighted BCEWithLogitsLoss** was used (use_weighted_loss=True). Class weights (pos_weight) were calculated based on the inverse frequency of each technique in the training set, capped at a maximum value of 10.0 to prevent extreme weights for very rare classes.
        
    - **Label Smoothing:** A small amount of label smoothing (label_smoothing=0.05) was applied to prevent the model from becoming overconfident.
        
    - Focal Loss was considered (code includes FocalLoss class and use_focal_loss flag set to False), but the weighted BCE approach with label smoothing yielded the best results in validation.
        
    
    ```
    # Inside train_epoch function
    pos_weight_tensor = None
    if not CONFIG['use_focal_loss'] and CONFIG['use_weighted_loss'] and class_weights:
        weights = [class_weights.get(tech, 1.0) for tech in TECHNIQUES]
        pos_weight_tensor = torch.tensor(weights, dtype=torch.float).to(device)
        # ... setup criterion
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        # ... other loss options
        criterion = nn.BCEWithLogitsLoss() # Default if no weighting
    
    # Applying label smoothing
    if CONFIG['label_smoothing'] > 0:
        smoothed_targets = targets * (1 - CONFIG['label_smoothing']) + (CONFIG['label_smoothing'] / len(TECHNIQUES)) # Corrected smoothing application for multi-label
        loss = criterion(outputs, smoothed_targets)
    else:
        loss = criterion(outputs, targets)
    ```
    
4. **Data Augmentation:**
    
    - Simple random word deletion was applied to a subset (augment_ratio=0.2) of training samples each epoch. This involved randomly removing 20% of the words in a text, aiming to make the model less reliant on specific keywords.
        
    
    ```
    def data_augmentation(text, techniques):
        # Simplified example shown, original code had 50% chance then deletion
        if random.random() < CONFIG["augment_ratio"]: # Augment based on ratio directly
             words = text.split()
             if len(words) > 5:
                 indices_to_delete = random.sample(range(len(words)), int(len(words) * 0.2))
                 words = [word for i, word in enumerate(words) if i not in indices_to_delete]
                 text = ' '.join(words)
        # Original code returned techniques, adjust if necessary for pipeline
        return text # Simplified return for clarity
    ```
    
5. **Threshold Optimization:**
    
    - Since Macro F1 is sensitive to the classification threshold (especially in multi-label settings), optimal thresholds were determined per technique after each validation epoch where performance improved.
        
    - This involved iterating through potential thresholds (0.15 to 0.85) for each technique on the validation set predictions and selecting the threshold that maximized the individual F1 score for that technique.
        
    - The best set of thresholds found during training (corresponding to the best validation Macro F1 epoch) was saved with the model checkpoint and used for generating the final test predictions. This step was critical for maximizing the competition metric.
        
    
    ```
    def find_optimal_threshold(y_true, y_pred_proba):
        thresholds = {}
        logger.info("Finding optimal thresholds...")
        for i, technique in enumerate(TECHNIQUES):
            best_f1 = 0
            best_threshold = 0.5
            true_labels = y_true[:, i]
            pred_probs = y_pred_proba[:, i]
            for threshold in np.arange(0.15, 0.85, 0.01):
                preds = (pred_probs >= threshold).astype(int)
                f1 = f1_score(true_labels, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            thresholds[technique] = best_threshold
        return thresholds
    
    # Usage during evaluation and prediction
    # val_results = evaluate(...)
    # if "optimal_thresholds" in val_results and val_f1 > best_f1:
    #     thresholds = val_results["optimal_thresholds"]
    # ... save thresholds with best model ...
    # submission_df = predict_test_data(model, tokenizer, test_path, thresholds)
    ```
    
6. **What Didn't Work (or wasn't the final choice):**
    
    - **Focal Loss:** While implemented, it didn't outperform the weighted BCEWithLogitsLoss + label smoothing combination in validation experiments for the final model configuration.
        
    - **Default Threshold (0.5):** Using a fixed 0.5 threshold for all classes yielded significantly lower Macro F1 scores compared to the optimized per-class thresholds.
        

**Model Inference**

For the final submission, the best model checkpoint (saved based on the highest validation Macro F1 score) was loaded. The corresponding optimal thresholds saved with that checkpoint were used. The predict_test_data function processed the test data, generated predictions using the loaded model, applied the optimal thresholds to convert probabilities to binary labels (0 or 1) for each technique, and formatted the results into the required submission.csv file.

```
def predict_test_data(model, tokenizer, test_file, thresholds=None):
    logger.info(f"Loading test data from {test_file}")
    test_df = process_data(test_file) # Ensure consistent preprocessing

    test_dataset = ManipulationDataset(
        texts=test_df['content'].values,
        tokenizer=tokenizer,
        max_length=CONFIG['max_length']
    )
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], ...)

    # Evaluate function returns probabilities and thresholded predictions
    results = evaluate(model, test_loader, CONFIG['device'], thresholds)
    predictions = results["predictions"] # Uses the optimized thresholds passed in

    submission_df = pd.DataFrame()
    submission_df['id'] = test_df['id']
    for i, technique in enumerate(TECHNIQUES):
        submission_df[technique] = [pred[i] for pred in predictions]

    return submission_df

# Main execution flow
# ... train_model returns best thresholds ...
# checkpoint = torch.load('best_model.pt')
# model.load_state_dict(checkpoint['model_state_dict'])
# thresholds = checkpoint['thresholds'] # Load best thresholds
# submission_df = predict_test_data(model, tokenizer, test_path, thresholds)
# submission_df.to_csv('submission.csv', index=False)
```

**Conclusion**

The success of this solution stemmed from combining a powerful pre-trained model (xlm-roberta-large) with careful fine-tuning strategies. Key elements included the custom head with MCDropout-like averaging, addressing class imbalance through weighted loss and label smoothing, applying simple data augmentation, and critically, optimizing the decision thresholds per technique based on the validation set performance.

**Sources**

- **Model:** XLM-RoBERTa - [Hugging Face Model Card](https://www.google.com/url?sa=E&q=https%3A%2F%2Fhuggingface.co%2Fxlm-roberta-large)
    
- **Libraries:**
    
    - Hugging Face Transformers: [https://huggingface.co/docs/transformers/index](https://www.google.com/url?sa=E&q=https%3A%2F%2Fhuggingface.co%2Fdocs%2Ftransformers%2Findex)
        
    - PyTorch: [https://pytorch.org/](https://www.google.com/url?sa=E&q=https%3A%2F%2Fpytorch.org%2F)
        
    - Scikit-learn: [https://scikit-learn.org/stable/](https://www.google.com/url?sa=E&q=https%3A%2F%2Fscikit-learn.org%2Fstable%2F)
        

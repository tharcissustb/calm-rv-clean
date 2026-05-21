def train_one_fold(train_rows, val_rows, test_rows, test_event_name, args, cfg, device):
    """
    Train a single LOEO fold with optional adaptive calibration
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["name"],
        num_labels=cfg["model"]["num_labels"],
    )
    model.to(device)

    train_ds = RumourDataset(train_rows, tokenizer, cfg["model"]["max_length"])
    val_ds = RumourDataset(val_rows, tokenizer, cfg["model"]["max_length"])
    test_ds = RumourDataset(test_rows, tokenizer, cfg["model"]["max_length"])

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]))

    total_steps = len(train_loader) * cfg["training"]["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    base_lambda_cal = cfg.get("calibration", {}).get("lambda_cal", 0.5)

    best_f1 = -1.0
    best_state = None
    
    contrastive_loss_fn = ContrastiveAlignmentLoss(temperature=0.07)

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            texts = batch.get("text", [])
            batch_events = batch.get("event_id", [])
            
            # Remove non-tensor fields
            batch.pop("id", None)
            batch.pop("text", None)
            batch.pop("event_id", None)
            
            labels = batch.pop("labels").to(device)
            
            # Only move tensor values to device
            input_batch = {}
            for k, v in batch.items():
                if hasattr(v, 'to'):
                    input_batch[k] = v.to(device)
            
            optimizer.zero_grad()
            outputs = model(**input_batch, labels=labels)
            loss = outputs.loss

            # Get representations for contrastive loss
            reps = get_representations(model, input_batch["input_ids"], input_batch["attention_mask"])

            # L_align: Contrastive alignment
            if args.use_align:
                align_loss = contrastive_loss_fn(reps, labels, batch_events)
                loss = loss + args.lambda_align * align_loss

            # L_cal: Calibration loss
            if args.use_calibration:
                probs = F.softmax(outputs.logits, dim=-1)
                one_hot = F.one_hot(labels, num_classes=3).float()
                cal_loss = torch.mean((probs - one_hot) ** 2)
                loss = loss + base_lambda_cal * cal_loss

            # L_robust: Perturbation robustness
            if args.use_robust:
                perturbed = [simple_perturb(t) for t in texts]
                pert_enc = tokenizer(perturbed, padding=True, truncation=True, max_length=128, return_tensors="pt")
                pert_input = {}
                for k, v in pert_enc.items():
                    if hasattr(v, 'to'):
                        pert_input[k] = v.to(device)
                pert_outputs = model(**pert_input)
                robust_loss = F.cross_entropy(pert_outputs.logits, labels)
                loss = loss + args.lambda_robust * robust_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        all_labels, all_logits = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch.pop("id", None)
                batch.pop("text", None)
                batch.pop("event_id", None)
                labels = batch.pop("labels").numpy()
                
                input_batch = {}
                for k, v in batch.items():
                    if hasattr(v, 'to'):
                        input_batch[k] = v.to(device)
                
                outputs = model(**input_batch)
                logits = outputs.logits.cpu().numpy()
                all_labels.extend(labels)
                all_logits.append(logits)

        all_logits = np.vstack(all_logits)
        val_metrics = classification_metrics(all_logits, np.array(all_labels))

        print(f"Epoch {epoch+1} | loss={avg_loss:.4f} | val_f1={val_metrics['macro_f1']:.4f} | val_ece={val_metrics['ece']:.4f}")

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test
    model.eval()
    all_labels, all_logits = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch.pop("id", None)
            batch.pop("text", None)
            batch.pop("event_id", None)
            labels = batch.pop("labels").numpy()
            
            input_batch = {}
            for k, v in batch.items():
                if hasattr(v, 'to'):
                    input_batch[k] = v.to(device)
            
            outputs = model(**input_batch)
            logits = outputs.logits.cpu().numpy()
            all_labels.extend(labels)
            all_logits.append(logits)

    all_logits = np.vstack(all_logits)
    test_metrics = classification_metrics(all_logits, np.array(all_labels))

    return test_metrics
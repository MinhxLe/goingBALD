from sklearn.metrics import f1_score
import tqdm

def epoch_run(
    model,
    data_loader,
    criterion,
    score_fun=None,
    trainer_mode=False,
    optimizer = None,
    ):
    """
    Boilerplate code for running an epoch.
    I am following pytorch and using input/target as names.

    Args
    ----
    model: pytorch model
    data_loader: pytorch Dataloader to load data
    criterion: the function to compute loss
        it should take as args "input" and "target"
    score_fun: function to compute score
        it should take as args "input" and "target"
        if None, score_fun will be set to criterion
    trainer_mode: if True, will update model parameters
    optimizer: optimizer to use to update model parameters
    """
    model.eval()
    if trainer_mode is True:
        assert optimizer is not None
        model.train()

    if score_fun is None:
        score_fun = criterion

    losses = []
    scores = []
    with tqdm.tqdm(total=len(data_loader)) as progress_bar:
        for datum in data_loader:

            x_in, y_target = datum
            y_pred = model(x_in)

            loss = criterion(input=y_pred,target=y_target)
            losses.append(loss.item())

            score = score_fun(input=y_pred,target=y_target)
            scores.append(score)

            progress_bar.update(1)

            if trainer_mode is True:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    avg_loss = sum(losses)/len(losses)
    avg_score = sum(scores)/len(scores)
    return {"loss":avg_loss, "score":avg_score}

def train(n_epochs, model, dataloder, optimizer, loss_function, device):
    model.train()
    LOSSES = []
    for epoch in range(n_epochs):
        model.train()
        dataloder = dataloder
        losses = []
        for x, t in dataloder:
            x, t = x.to(device), t.to(device)
            optimizer.zero_grad()
            y = model(x)
            loss = loss_function(y,t)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        LOSSES.append(sum(losses) / len(losses))
    return LOSSES    

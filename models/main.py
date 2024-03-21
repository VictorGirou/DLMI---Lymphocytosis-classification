"""
"""
import pandas as pd 
import torch
import numpy as np 

from sklearn.preprocessing import MinMaxScaler
from DeepSets import DeepSets, deepsets_loader


def main():
    """
    """
    # download data
    df = pd.read_csv("./../data/resnet50train.csv")
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df_test = pd.read_csv("./../data/resnet50test.csv")
    df_test.drop('Unnamed: 0', axis=1, inplace=True)

    df_ann = pd.read_csv("./../data/clinical_annotation.csv")
    df_ann.set_index("ID", inplace=True)

    # join datasets 
    df = df.join(df_ann.LABEL, on="ID", how="left")
    df_test = df_test.join(df_ann.LABEL, on="ID", how="left")

    # scaling 
    col = [str(i) for i in range(2048)]
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[col]), columns=col)
    df_test_scaled = pd.DataFrame(scaler.transform(df_test[col]), columns=col)

    df_scaled["ID"] = df.ID
    df_scaled["LABEL"] = df.LABEL
    df_test_scaled["ID"] = df_test.ID
    df_test_scaled["LABEL"] = df_test.LABEL

    # print(df_scaled.isna().sum())

    # construct data loader
    loader = deepsets_loader(df_scaled)
    test_loader = deepsets_loader(df_test_scaled)

    # define model 
    model = DeepSets(2048, 32, 1)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Assuming custom_dataset and custom_dataloader are defined as in the previous example

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in loader:
            inputs, labels = batch
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute the loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Optimize
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(loader)}")

        # Evaluate the model
        total = 0
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                input, labels = data
                output = model(input)
                predictions = (output > 0.5).float()  # Convert probabilities to binary predictions
                total += len(data)
                correct += (predictions == labels).sum().item() 

        
    
        # Calculate accuracy
        accuracy = correct / total

        print("Accuracy: ", accuracy, "\n")

    print("Training finished")


if __name__ == "__main__":
    main()
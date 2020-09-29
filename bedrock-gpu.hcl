version = "1.0"

train {
    step train {
        image = "python:3.7"
        install = ["pip3 install -r requirements.txt"]
        script = [
            {
                sh = ["python3 train.py"]
            }
        ]
        
        resources {
            cpu = "5G"
            memory = "50G"
            gpu = "1"
        }
    }
    
    
    
    parameters {
        HIDDEN_DIM = "512"
        DROPOUT_1 = "0.5"
        DROPOUT_2 = "0.5"
        N_EPOCH = "10"
        BATCH_SIZE = "32"
    }
}

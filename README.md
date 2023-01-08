# Simple Federated Learning using gRPC

## Install python libraries
```
$ cd sync_fl
$ pip3 install requirements.txt
```

## Getting started
To run workers, open new terminals and run the  following command: `python3 src/server + port` 

Sample command:
```
// On terminal 1
python3 src/node.py 8081
// On terminal 2
python3 src/node.py 8082
```

To run CLI to play with these worker, open `src/connect.py` and config:
```
addresses = ['localhost:8081', 'localhost:8082']
```
Then run following command to play around
```
// On terminal 3
python3 src/client.py
```


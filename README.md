# ecg-arrhythmia-classifier

Visualize waveforms
https://www.physionet.org/lightwave/?db=mitdb/1.0.0 


For the dataset:
'
wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/
aws s3 sync --no-sign-request s3://physionet-open/mitdb/1.0.0/ DESTINATION
'
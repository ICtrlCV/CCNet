#! /bin/bash
sudo netstat -tunpl | grep 12345
sleep 1s
kill -9 $(ps -ef | grep "socket2springboot.py" | grep -v grep | awk "{print $2}")
sleep 1s
nohup python -u tools/socket2springboot.py > /home/ubuntu/cy/net/log/socket2springboot.log 2>&1 &
sleep 1s
ps -def | grep "socket2springboot.py"
sleep 1s
python tools/method_test.py
echo "wait inference"
sleep 2s
echo "inference finish"
vim log/socket2springboot.log
#! /bin/bash
time_stamp=$(date +"%Y%m%d%H%M%S")
project_path="/your/project/path/CCNet"
sudo netstat -tunpl | grep 12345
sleep 1s
kill $(ps -ef | grep "socket2springboot.py" | grep -v grep | awk "{print $2}")
sleep 1s
echo "run socket2springboot.py"
nohup python -u tools/socket2springboot.py > "${project_path}/log/socket2springboot_${time_stamp}.log" 2>&1 &
sleep 1s
echo "check socket2springboot.py process"
ps -def | grep "socket2springboot.py"
sleep 1s
echo "run test function"
python test/method_test.py
echo "wait inference"
sleep 2s
echo "inference finish"
vim log/socket2springboot_${time_stamp}.log
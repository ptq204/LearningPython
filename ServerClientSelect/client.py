import sys
import socket
import os

s = socket.socket()

host = socket.gethostname()
port = 5000
print("client will be connected to {}".format(host))
s.connect(('127.0.0.1',port))
print("connected to server")

while(1):
    incoming_msg = s.recv(1024)
    incoming_msg = incoming_msg.decode()
    print("server: {}".format(incoming_msg))
    msg = input("your: ")
    msg = msg.encode()
    s.send(msg)
    print("sent")
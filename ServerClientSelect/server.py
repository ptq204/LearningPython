import socket, select
s = socket.socket()
host = socket.gethostname()
port = 5000
s.bind(('127.0.0.1', port))
s.listen(5)
inputs = [s]

while True:
    rs, ws, es = select.select(inputs, [], [])
    for r in rs:
        if r is s:
            cnn,addr = s.accept()
            addrclient = addr
            print("Got connection from {}".format(addr))
            inputs.append(cnn)
            msg = input("server: ")
            msg = msg.encode()
            cnn.send(msg)
        else:
            try:
                data = r.recv(1024)
                data = data.decode()
                disconnected = not data
            except socket.error:
                disconnected = True
            if disconnected:
                print(r.getpeername() + 'disconnected')
                inputs.remove(r)
            else:
                print(data)
                msg = input("server: ")
                msg = msg.encode()
                cnn.send(msg)


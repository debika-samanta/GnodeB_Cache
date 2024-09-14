# My Project

Here is an overview of architechture of project.

![Project Diagram](/architecture.png)

## System Architecture

The following diagram illustrates the architecture where:

- **UE (Client)**: User Equipment requesting data.
- **gNodeB**: Base station that communicates with the UE and contains a cache proxy.
- **Cache Proxy**: Caching component running on the gNodeB.
- **Data Network (DN)**: The server providing the requested data.


Deploy the core network
>> oai-cn5g-fed/docker-compose$ docker-compose -f docker-compose-basic-vpp-nrf.yaml up -d

For running 5G RAN using UERANSIM 
>> git clone -b docker_support https://github.com/orion-belt/UERANSIM.git

>> cd UERANSIM
>>  docker build --target ueransim --tag ueransim:latest -f docker/Dockerfile.ubuntu.18.04 .

>> oai-cn5g-fed/docker-compose$ docker-compose -f docker-compose-ueransim-vpp.yaml up -d

To check the logs
>> docker logs ueransim

Ping UE from external DN container.

>> docker exec -it oai-ext-dn ping -c 3 12.2.1.2

# nat port forwarding 
>> sudo sysctl -w net.ipv4.ip_forward=1

>> sudo iptables -t nat -A POSTROUTING -o enp0s3 -j MASQUERADE

>> sudo systemctl stop ufw

>> sudo iptables -I FORWARD 1 -j ACCEPT

Network overview 

       +-----------------+
       |                 |
       |        UE        |
       |  (Client)        |
       |                 |
       +--------+--------+
                |
                | HTTP/UDP
                |
                v
       +--------+--------+
       |                 |
       |     gNodeB       |
       |   +---------+    |
       |   |  Proxy  |    |
       |   | (Cache) |    |
       |   +----+----+    |
       |        |         |
       +--------+---------+
                |
                | HTTP/UDP
                |
                v
       +--------+--------+
       |                 |
       |        DN        |
       |   (Server)       |
       |                 |
       +-----------------+

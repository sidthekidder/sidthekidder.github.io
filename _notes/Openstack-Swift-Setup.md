---
layout: post
title: OpenStack Swift Setup
type: misc
date: December 9, 2019
---

Swift is the object storage service included in OpenStack's environment. To setup swift, you first need to setup the OpenStack environment and the Keystone Identity service.

[Install OpenStack Environment](https://docs.openstack.org/install-guide/environment.html)

[Install Keystone Identity Service](https://docs.openstack.org/keystone/rocky/install/index.html)

[Install Swift Object Storage](https://docs.openstack.org/swift/rocky/install/)


The hardest part of this process was getting the network configuration right. I used several interconnected VMs on different physical machines to host the controller and storage nodes. Make sure that the VMs are able to ping each other (using a bridge network) and also able to ping the internet (using a NAT), and that they aren't blocked by any firewalls.

### Architecture Overview
![network architecture overview](/images/notes/openstackswift/config.png)


### Verification

Here's a quick list of common commands to verify the system was successfully installed -

Show the system status:

`swift stat`

Create a container:

`openstack container create container1`

List all containers in the current account:

`swift list `

Delete a container:

`swift delete container1`

Create/upload objects in a container:

`openstack object create container1 FILE`

`swift upload container1 FILE`

List all the objects in a container:

`openstack object list container1`

`swift list container1`

Download an object:

`openstack object save container1 FILE`

`swift download FILE`

Delete an object:

`swift delete container1 FILE`

Find an object in disk (CentOS 7): 

Check the location of sdb and sdc using the command: `df -lh` (in /srv/node/sdb for me)


### Load-Testing

If you want to carry out basic load tests, this is a nice tool - [https://github.com/christianbaun/ossperf](https://github.com/christianbaun/ossperf). The ossperf performance analysis tool creates a user-defined number of files with random content and of a specified size inside a local directory. The tool creates a container, uploads and downloads the files, and afterward removes the container. The time required to carry out these tasks is measured and printed out on the command line.

Load testing experiment parameters:

- `./ossperf.sh -n 5 -s 4096 -a -k -p -o`
- number of files to be created: 5
- size of each file: 4096 KB
- parallel upload and download enabled

![load testing results](/images/notes/openstackswift/loadtest.png)


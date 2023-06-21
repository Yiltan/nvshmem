/*
 * Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "sockets.h"

int ipcOpenSocket(ipcHandle *&handle) {
    int sock = 0;
    struct sockaddr_un cliaddr;

    handle = new ipcHandle;
    memset(handle, 0, sizeof(*handle));

    if ((sock = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0) {
        perror("IPC failure:Socket creation error");
        delete handle;
        return -1;
    }

    bzero(&cliaddr, sizeof(cliaddr));
    cliaddr.sun_family = AF_UNIX;
    char temp[50];

    // Create unique name for the socket.
    int name_len = snprintf(temp, 50, "/tmp/nvshmem-socket-%u", getpid());
    if (name_len < 0 || name_len >= 50) {
        printf("Error formatting socket file name\n");
        delete handle;
        close(sock);
        return -1;
    }

    strncpy(cliaddr.sun_path, temp, 50);
    if (bind(sock, (struct sockaddr *)&cliaddr, sizeof(cliaddr)) < 0) {
        perror(
            "IPC failure: Binding socket failed. If you have any (stale) files"
            "with names like /tmp/nvshmem-socket-<0-9>*, delete or rename them!");
        delete handle;
        close(sock);
        return -1;
    }

    handle->socket = sock;
    handle->socketName = new char[strlen(temp) + 1];
    strcpy(handle->socketName, temp);

    return 0;
}

int ipcCloseSocket(ipcHandle *handle) {
    if (!handle) {
        return -1;
    }

    if (handle->socketName) {
        unlink(handle->socketName);
        delete[] handle->socketName;
    }
    close(handle->socket);
    delete handle;
    return 0;
}

int ipcRecvFd(ipcHandle *handle, int *shHandle) {
    struct msghdr msg = {0, 0, 0, 0, 0, 0, 0};
    struct iovec iov[1];

    // Union to guarantee alignment requirements for control array
    union {
        struct cmsghdr cm;
        char control[CMSG_SPACE(sizeof(int))];
    } control_un;

    struct cmsghdr *cmptr;
    int receivedfd;
    char dummy_buffer[1];

    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);

    iov[0].iov_base = (void *)dummy_buffer;
    iov[0].iov_len = sizeof(dummy_buffer);

    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    if (recvmsg(handle->socket, &msg, 0) <= 0) {
        perror("IPC failure: Receiving data over socket failed");
        return -1;
    }

    if (((cmptr = CMSG_FIRSTHDR(&msg)) != NULL) && (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
        if ((cmptr->cmsg_level != SOL_SOCKET) || (cmptr->cmsg_type != SCM_RIGHTS)) {
            return -1;
        }

        memmove(&receivedfd, CMSG_DATA(cmptr), sizeof(receivedfd));
        *(int *)shHandle = receivedfd;
    } else {
        return -1;
    }

    return 0;
}

int ipcSendFd(ipcHandle *handle, const int shareableHandle, pid_t process) {
    struct msghdr msg;
    struct iovec iov[1];

    union {
        struct cmsghdr cm;
        char control[CMSG_SPACE(sizeof(int))];
    } control_un;

    struct cmsghdr *cmptr;
    struct sockaddr_un cliaddr;

    // Construct client address to send this SHareable handle to
    bzero(&cliaddr, sizeof(cliaddr));
    cliaddr.sun_family = AF_UNIX;
    char temp[50];
    int name_len = snprintf(temp, 50, "/tmp/nvshmem-socket-%u", process);
    if (name_len < 0 || name_len >= 50) {
        printf("Error formatting socket file name\n");
        return -1;
    }
    strncpy(cliaddr.sun_path, temp, 50);

    // Send corresponding shareable handle to the client
    int sendfd = (int)shareableHandle;

    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);

    cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;

    memmove(CMSG_DATA(cmptr), &sendfd, sizeof(sendfd));

    msg.msg_name = (void *)&cliaddr;
    msg.msg_namelen = sizeof(struct sockaddr_un);

    iov[0].iov_base = (void *)"";
    iov[0].iov_len = 1;
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    msg.msg_flags = 0;

    ssize_t sendResult = sendmsg(handle->socket, &msg, 0);
    if (sendResult <= 0) {
        perror("IPC failure: Sending data over socket failed");
        return -1;
    }

    return 0;
}

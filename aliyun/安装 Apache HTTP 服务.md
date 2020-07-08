## 安装 Apache HTTP 服务

Apache是世界使用排名第一的Web服务器软件。它可以运行在几乎所有广泛使用的计算机平台上，由于其跨平台和安全性被广泛使用，是最流行的Web服务器端软件之一。

\1.  执行如下命令，安装Apache服务及其扩展包。

```
yum -y install httpd httpd-manual mod_ssl mod_perl mod_auth_mysql
```

返回类似如下图结果则表示安装成功。

![img](https://img.alicdn.com/tfs/TB1l7DUHpY7gK0jSZKzXXaikpXa-1050-137.png)



\2.  执行如下命令，启动Apache服务。

```
systemctl start httpd.service
```

\3.  测试Apache服务是否安装并启动成功。

Apache默认监听80端口，所以只需在浏览器访问ECS分配的IP地址http://<ECS公网地址>，如下图：



![img](https://img.alicdn.com/tfs/TB1HmVpaepyVu4jSZFhXXbBpVXa-1920-937.png)
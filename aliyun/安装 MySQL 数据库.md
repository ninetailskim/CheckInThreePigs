## 安装 MySQL 数据库

由于使用wordpress搭建云上博客，需要使用MySQL数据库存储数据，所以这一步我们安装一下MySQL。

\1.  执行如下命令，下载并安装MySQL官方的`Yum Repository`。

```
wget http://dev.mysql.com/get/mysql57-community-release-el7-10.noarch.rpm
yum -y install mysql57-community-release-el7-10.noarch.rpm
yum -y install mysql-community-server
```



![img](https://img.alicdn.com/tfs/TB1BRnVHxz1gK0jSZSgXXavwpXa-958-431.png)

\2.  执行如下命令，启动 MySQL 数据库。

```
systemctl start mysqld.service
```

\3.  执行如下命令，查看MySQL运行状态。

```
systemctl status mysqld.service
```



![img](https://img.alicdn.com/tfs/TB1gszWHuT2gK0jSZFvXXXnFXXa-945-229.png)

\4.  执行如下命令，查看MySQL初始密码。

```
grep "password" /var/log/mysqld.log
```



![img](https://img.alicdn.com/tfs/TB1FmNpaepyVu4jSZFhXXbBpVXa-834-36.png)

\5.  执行如下命令，登录数据库。

```
mysql -uroot -p
```



![img](https://img.alicdn.com/tfs/TB1Wz6UHvb2gK0jSZK9XXaEgFXa-675-226.png)

\6.  执行如下命令，修改MySQL默认密码。

> **说明** 新密码设置的时候如果设置的过于简单会报错，必须同时包含大小写英文字母、数字和特殊符号中的三类字符。

```
ALTER USER 'root'@'localhost' IDENTIFIED BY 'NewPassWord1.';
```

\7.  执行如下命令，创建wordpress库。

```
create database wordpress; 
```

\8.  执行如下命令，创建wordpress库。 执行如下命令，查看是否创建成功。

```
show databases;
```

\9.  输入`exit`退出数据库。



![img](https://img.alicdn.com/tfs/TB14Bj0HxD1gK0jSZFKXXcJrVXa-836-664.png)
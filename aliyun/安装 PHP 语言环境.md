## 安装 PHP 语言环境

WordPress是使用PHP语言开发的博客平台，用户可以在支持PHP和MySQL数据库的服务器上架设属于自己的网站。也可以把WordPress当作一个内容管理系统（CMS）来使用。

\1.  执行如下命令，安装PHP环境。

```
yum -y install php php-mysql gd php-gd gd-devel php-xml php-common php-mbstring php-ldap php-pear php-xmlrpc php-imap
```

\2.  执行如下命令创建PHP测试页面。

```
echo "<?php phpinfo(); ?>" > /var/www/html/phpinfo.php
```

\3.  执行如下命令，重启Apache服务。

```
systemctl restart httpd
```

\4.  打开浏览器，访问`http://<ECS公网地址>/phpinfo.php`，显示如下页面表示PHP语言环境安装成功。

![img](https://img.alicdn.com/tfs/TB1oCVpaepyVu4jSZFhXXbBpVXa-601-840.png)
## Wordpress安装和配置

本小节将在已搭建好的LAMP 环境中，安装部署 WordPress

\1.  执行如下命令，安装wordpress。

```
yum -y install wordpress
```

显示如下信息表示安装成功。

![img](https://img.alicdn.com/tfs/TB1b02VHEz1gK0jSZLeXXb9kVXa-1042-156.png)

\2.  修改WordPress配置文件。

1）执行如下命令，修改wp-config.php指向路径为绝对路径。

```
# 进入/usr/share/wordpress目录。
cd /usr/share/wordpress
# 修改路径。
ln -snf /etc/wordpress/wp-config.php wp-config.php
# 查看修改后的目录结构。
ll
```

2）执行如下命令，移动wordpress到Apache根目录。

```
# 在Apache的根目录/var/www/html下，创建一个wp-blog文件夹。
mkdir /var/www/html/wp-blog
mv * /var/www/html/wp-blog/
```

3）执行以下命令修改wp-config.php配置文件。

在执行命令前，请先替换以下三个参数值。

- database_name_here为之前步骤中创建的数据库名称，本示例为wordpress。
- username_here为数据库的用户名，本示例为root。
- password_here为数据库的登录密码，本示例为NewPassWord1.。

```
sed -i 's/database_name_here/wordpress/' /var/www/html/wp-blog/wp-config.php
sed -i 's/username_here/root/' /var/www/html/wp-blog/wp-config.php
sed -i 's/password_here/NewPassWord1./' /var/www/html/wp-blog/wp-config.php
```

4）执行以下命令，查看配置文件信息是否修改成功。

```
cat -n /var/www/html/wp-blog/wp-config.php
```



![img](https://img.alicdn.com/tfs/TB1j02VHEz1gK0jSZLeXXb9kVXa-913-631.png)

\3.  执行如下命令，重启Apache服务。

```
systemctl restart httpd
```
## Обновление Python до версии 3.11 в WSL 2 Ubuntu

**1. Добавление PPA Deadsnakes:**

```
sudo add-apt-repository ppa:deadsnakes/ppa
```

**2. Обновление списка пакетов:**

```
sudo apt update
```

**3. Установка Python 3.11:**

```
sudo apt install python3.11
```

**4. Проверка версии Python:**

```
python3 --version
```

**Дополнительно:**

* **Установка pip3 для Python 3.11:**

```
sudo apt install python3.11-pip
```

* **Создание символической ссылки для Python 3.11:**

```
sudo ln -s /usr/bin/python3.11 /usr/bin/python
```

**Примечание:**

* После обновления Python 3.10 до 3.11, Python 3.10 будет доступен как `python3.10`.
* Вы можете установить несколько версий Python одновременно.

**Ссылки:**

* Установка Python 3.11 в Ubuntu и других дистрибутивах Linux: [https://www.debugpoint.com/install-python-3-11-ubuntu/](https://www.debugpoint.com/install-python-3-11-ubuntu/)
* PPA Deadsnakes: [https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa)

FROM pypy:2

RUN pip install --upgrade pip
RUN pip install enum34

RUN git clone https://github.com/markovi/PyClick.git /tmp/PyClick
RUN cd /tmp/PyClick && pypy setup.py install

ADD . /usr/src/myapp

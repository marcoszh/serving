FROM marcoszh/inception_serving:latest

ADD entrypoint.sh /opt/

CMD ["/opt/entrypoint.sh"]
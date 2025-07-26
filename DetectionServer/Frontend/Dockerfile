FROM nginx:latest

WORKDIR /app/Frontend
COPY . /app/Frontend

COPY default.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
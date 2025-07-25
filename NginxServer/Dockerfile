FROM nginx:latest

WORKDIR /app

COPY ./images /app/WebGallery/images
COPY ./index.html /app/WebGallery/index.html
COPY ./style.css /app/WebGallery/style.css
COPY ./default.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

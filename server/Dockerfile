FROM node:22-slim

WORKDIR /app
COPY package.json ./
RUN npm install --omit=dev

COPY server.js ./

ENV UPLOAD_DIR=/data/incoming \
    PORT=8080

EXPOSE 8080
CMD ["node", "server.js"]

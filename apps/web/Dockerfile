FROM node:20-alpine

WORKDIR /usr/src/app
ENV NEXT_TELEMETRY_DISABLED=1

COPY package.json package-lock.json* .npmrc* ./
RUN npm install

COPY . .
EXPOSE 3000
CMD ["npm","run","dev"]

# Markdown capabilities

1. Can include a regular markdown file from anywhere in your repository
2. Can specify where to publish the site
3. Can  generate a `ssg.json` with a docker command 
   
```docker run --rm -v $PWD:/work us-docker.pkg.dev/swimmio/public/swimm-ssg:0 init -C /work```

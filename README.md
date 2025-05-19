# Evaluating Cluster

## Architecture Diagrams
![Architecture Diagram](./assets/diagrams.png)


Create code to evaluate

The evaluation dataset must be in the following format

```json
[
    {
        "id":1,
        "question": "1 + 1 = ?",
        "choices" : [
            "A. 1",
            "B. 2",
            "C. 3",
            "D. 4"
        ],
        "answer": "D"
    }
]
```

Test
```bash
python -m test.test
```

Remenber add env file
```bash
WANDB_API_KEY=your_api_key
WANDB_PROJECT=mlops
WANDB_ENTITY=neu-solution
```

## Run docker evaluation 
```bash
# Build the Docker image

# evaluation function
docker build -t evaluate_model -f Dockerfile.eval .

# evaluation server
docker build -t evaluate_model -f Dockerfile .

# Run the container with tests
docker run --gpus all --env-file .env -v ~/.cache/huggingface:/root/.cache/huggingface  evaluate_model 
```

## Run docker api service
```bash
docker build -t evaluation-api .

docker run --gpus all --env-file .env -p 23477:23477 -v ~/.cache/huggingface:/root/.cache/huggingface evaluation-api
```

## Run via docker compose
```bash
docker-compose up --build -d
```

### Built With

This section lists major frameworks/libraries used to bootstrap the project:

![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue) ![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-017CEE?style=for-the-badge&logo=Apache%20Airflow&logoColor=white) ![bucket][S3] ![db][RDS] ![dbms][pg] ![ui][Next.js] [![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?logo=github-actions&logoColor=white)](#) [![Vercel](https://img.shields.io/badge/Vercel-%23000000.svg?logo=vercel&logoColor=white)](#) ![nVIDIA](https://img.shields.io/badge/cuda-000000.svg?style=for-the-badge&logo=nVIDIA&logoColor=green) ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi) [![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=fff)](#) 


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/mpquochung/chatbot_hr/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/mpquochung/chatbot_hr/blob/main/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/qhungmp/
[product-screenshot]: image/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[Streamlit]: https://img.shields.io/badge/Streamlit-%23FF4B4B?logo=streamlit&color=white
[Streamlit-url]: https://streamlit.io
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
[S3]: https://img.shields.io/badge/AWS%20S3-%23569A31?logo=amazons3&color=white
[RDS]: https://img.shields.io/badge/AWS%20RDS-%23569A31?logo=amazonrds&color=white
[EC2]: https://img.shields.io/badge/AWS%20EC2-%23569A31?logo=amazonec2&color=white
[OPENAI]: https://img.shields.io/badge/OPENAI%20API-%23412991?logo=openai&logoColor=%23412991&color=white
[Claude]: https://img.shields.io/badge/AWS%20Bedrock%20Claude3-%23191919?logo=anthropic&logoColor=%23191919&color=white
[pg]: https://img.shields.io/badge/Postgres%20SQL-%234169E1?logo=postgresql&logoColor=%234169E1&color=white
[architecture]: image/architecture.png
[vllm]: https://blog.vllm.ai/assets/logos/vllm-logo-only-light.png



# Spring AI Integration

[Spring AI](https://docs.spring.io/spring-ai/reference/) provides Spring-based
AI application development. Since DistLLM exposes an OpenAI-compatible API,
Spring AI's OpenAI client can connect to DistLLM with just a URL change.

## Quick Start

```xml
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-openai-spring-boot-starter</artifactId>
</dependency>
```

```properties
# application.properties
spring.ai.openai.base-url=http://localhost:8000/v1
spring.ai.openai.api-key=optional-key
spring.ai.openai.chat.model=llama-3-70b
```

```java
@RestController
public class ChatController {

    @Autowired
    private ChatClient chatClient;

    @GetMapping("/chat")
    public String chat(@RequestParam String message) {
        return chatClient.prompt()
            .user(message)
            .call()
            .content();
    }
}
```

## Configuration

| Property | Value |
|---|---|
| `spring.ai.openai.base-url` | `http://<distllm-host>:8000/v1` |
| `spring.ai.openai.chat.model` | Any model loaded on the cluster |
| `spring.ai.openai.api-key` | Required if DistLLM auth is enabled |

## Streaming

```java
Flux<String> stream = chatClient.prompt()
    .user("Tell me a story")
    .stream()
    .content();
```

No code changes needed beyond the URL — Spring AI's OpenAI client treats
DistLLM as a standard OpenAI-compatible endpoint.

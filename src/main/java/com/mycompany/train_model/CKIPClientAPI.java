package com.mycompany.train_model;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import com.google.gson.Gson;
import java.util.*;

public class CKIPClientAPI {
    // 修改為您的 Flask API URL；若服務部署在同一台機器，則可以使用 localhost
    private static final String API_URL = "http://localhost:5000/segment";

    /**
     * 單筆分詞：呼叫 API 取得分詞結果，回傳字串
     */
    public static String segment(String text) throws Exception {
        List<String> texts = Collections.singletonList(text);
        List<String> result = segmentBatch(texts);
        if (!result.isEmpty()) {
            return result.get(0);
        }
        return "";
    }

    /**
     * 批量分詞：傳入多筆文本，回傳分詞結果列表
     */
    public static List<String> segmentBatch(List<String> texts) throws Exception {
        HttpClient client = HttpClient.newHttpClient();
        Map<String, Object> payload = new HashMap<>();
        payload.put("texts", texts);
        String jsonPayload = new Gson().toJson(payload);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(new URI(API_URL))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonPayload))
                .build();

        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() == 200) {
            Map<?, ?> map = new Gson().fromJson(response.body(), Map.class);
            Object segmentedObj = map.get("segmented");
            if (segmentedObj instanceof List) {
                //noinspection unchecked
                return (List<String>) segmentedObj;
            }
        }
        throw new Exception("Error calling CKIP segmentation API: " + response.body());
    }
}

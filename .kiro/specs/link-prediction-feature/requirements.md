# Requirements Document

## Introduction

Tính năng dự báo liên kết (Link Prediction) cho dashboard phân tích mạng lưới tác giả. Khi người dùng focus vào một tác giả cụ thể, hệ thống sẽ hiển thị không chỉ các đồng tác giả hiện tại mà còn dự báo các tác giả có khả năng hợp tác trong tương lai, sử dụng 3 model đã được train (SBM, PA, PageRank) từ file `top3_models_optimized.pkl`.

## Glossary

- **Link Prediction**: Kỹ thuật dự đoán khả năng hình thành liên kết mới giữa hai node trong mạng lưới
- **SBM (Stochastic Block Model)**: Model dự đoán dựa trên cấu trúc cộng đồng của mạng
- **PA (Preferential Attachment)**: Model dự đoán dựa trên nguyên lý "rich get richer" - node có nhiều kết nối có xu hướng nhận thêm kết nối mới
- **PageRank**: Model dự đoán dựa trên độ quan trọng của node trong mạng
- **Ego Network**: Mạng con bao gồm một node trung tâm và tất cả các node kết nối trực tiếp với nó
- **Threshold**: Ngưỡng điểm số để quyết định một liên kết được dự đoán là "có khả năng xảy ra"

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to see predicted future collaborations when focusing on an author, so that I can identify potential research partnership opportunities.

#### Acceptance Criteria

1. WHEN a user focuses on an author THEN the Dashboard SHALL display both existing connections (solid lines) and predicted connections (dashed lines)
2. WHEN displaying predicted connections THEN the Dashboard SHALL use a visually distinct style (dashed lines with different color) to differentiate from existing connections
3. WHEN a predicted connection is shown THEN the Dashboard SHALL display the prediction confidence score from the best performing model (SBM)
4. WHEN the visualization loads THEN the Dashboard SHALL include a legend explaining the difference between solid lines (existing) and dashed lines (predicted)

### Requirement 2

**User Story:** As a user, I want to control the number of predicted connections displayed, so that I can avoid visual clutter while exploring the network.

#### Acceptance Criteria

1. WHEN in focus mode THEN the Dashboard SHALL provide a slider to control the maximum number of predicted connections (default: 5, range: 1-20)
2. WHEN predictions are generated THEN the Dashboard SHALL rank them by prediction score and display only the top N predictions
3. WHEN no valid predictions exist for an author THEN the Dashboard SHALL display an informative message indicating no predictions available

### Requirement 3

**User Story:** As a user, I want to see details about predicted connections in the sidebar, so that I can understand why certain collaborations are predicted.

#### Acceptance Criteria

1. WHEN in focus mode with predictions THEN the Dashboard SHALL display a separate table listing predicted collaborators with their prediction scores
2. WHEN displaying the predictions table THEN the Dashboard SHALL show author name and normalized prediction score (0-100%)
3. WHEN a user hovers over a predicted connection node THEN the Dashboard SHALL show a tooltip with the author name and prediction score

### Requirement 4

**User Story:** As a developer, I want the prediction data to be loaded efficiently, so that the dashboard remains responsive.

#### Acceptance Criteria

1. WHEN the dashboard starts THEN the System SHALL load the pickle file using Streamlit's caching mechanism
2. WHEN prediction scores need to be calculated THEN the System SHALL use the pre-computed model parameters from the pickle file
3. WHEN an author is not found in the prediction data THEN the System SHALL handle the error gracefully and continue displaying existing connections


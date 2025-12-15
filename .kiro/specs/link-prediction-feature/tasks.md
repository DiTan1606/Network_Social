# Implementation Plan

- [x] 1. Set up prediction data loading





  - [x] 1.1 Create cached function to load PKL file


    - Add `load_prediction_data()` function with `@st.cache_data` decorator
    - Handle FileNotFoundError gracefully
    - _Requirements: 4.1, 4.3_
  - [x] 1.2 Add prediction data loading call in main app flow

    - Load data at app startup alongside graph loading
    - Store in session or pass to functions as needed
    - _Requirements: 4.1_

- [x] 2. Implement SBM prediction logic





  - [x] 2.1 Create `calculate_sbm_score()` function


    - Input: two node IDs and sbm_data dict
    - Output: probability score from probs matrix
    - Handle missing block assignments gracefully
    - _Requirements: 4.2_
  - [ ]* 2.2 Write property test for SBM score calculation
    - **Property 4: SBM score calculation correctness**
    - **Validates: Requirements 4.2**
  - [x] 2.3 Create `get_link_predictions()` function


    - Calculate scores for all non-neighbor nodes
    - Filter out existing connections
    - Sort by score descending, return top N
    - _Requirements: 1.1, 2.2_
  - [ ]* 2.4 Write property test for predictions sorting and limiting
    - **Property 1: Predictions are sorted and limited**
    - **Validates: Requirements 2.2**
  - [ ]* 2.5 Write property test for excluding existing neighbors
    - **Property 2: Predictions exclude existing connections**
    - **Validates: Requirements 1.1**

- [x] 3. Implement score normalization





  - [x] 3.1 Create score normalization function


    - Normalize raw SBM scores to 0-100% range
    - Use min-max normalization based on threshold and max observed score
    - _Requirements: 3.2_
  - [ ]* 3.2 Write property test for score normalization bounds
    - **Property 3: Score normalization bounds**
    - **Validates: Requirements 3.2**

- [x] 4. Checkpoint - Ensure all tests pass





  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Update UI for predictions display





  - [x] 5.1 Add prediction controls to sidebar


    - Add slider for top_n predictions (range 1-20, default 5)
    - Only show when in focus mode
    - _Requirements: 2.1_
  - [x] 5.2 Add predicted nodes to visualization


    - Use distinct color (light purple) for predicted nodes
    - Add tooltip with author name and prediction score
    - _Requirements: 1.2, 3.3_
  - [x] 5.3 Add predicted edges with dashed style


    - Use dashed lines (pattern [5,5]) for predicted connections
    - Use purple color to distinguish from existing edges
    - _Requirements: 1.2, 1.3_
  - [ ]* 5.4 Write property test for edge styling attributes
    - **Property 5: Predicted edges have distinct styling**
    - **Validates: Requirements 1.2, 1.3**

- [x] 6. Add legend and predictions table





  - [x] 6.1 Add legend to visualization area


    - Explain solid lines = existing connections
    - Explain dashed lines = predicted connections
    - _Requirements: 1.4_
  - [x] 6.2 Add predictions table to sidebar


    - Display predicted author names with normalized scores
    - Show "No predictions available" when empty
    - _Requirements: 3.1, 3.2, 2.3_

- [x] 7. Final Checkpoint - Ensure all tests pass





  - Ensure all tests pass, ask the user if questions arise.


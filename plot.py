import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 변수 시각화
def draw_histogram(df, continuous_columns):
  total_plots = len(continuous_columns)
  cols = 2
  rows = math.ceil(total_plots / cols)
  plt.figure(figsize = (4 * cols, 3 * rows))

  for i, col in enumerate(continuous_columns, 1):
    plt.subplot(rows, cols, i)
    sns.histplot(x = df[col].dropna(), bins = 30, kde = True, color=sns.color_palette('pastel')[1])
    plt.title(f"Histogram of {col}")
    plt.tight_layout()
  plt.show()

# 시각화 함수 (이산형 변수 countplot)
def draw_countplot(df, categorical_columns):
    total_plots = len(categorical_columns)
    cols = 2
    rows = math.ceil(total_plots / cols)
    plt.figure(figsize=(4 * cols, 3 * rows))

    for i, col in enumerate(categorical_columns, 1):
      plt.subplot(rows, cols, i)

      # NaN 값을 포함한 경우, NaN을 제거한 후 카운트 정렬
      valid_values = df[col].dropna().astype(str)  # NaN 제거 후 문자열로 변환
      order_values = valid_values.value_counts().index  # 유효한 값 기준 정렬

      sns.countplot(x=valid_values, order=order_values, color=sns.color_palette('pastel')[1])
      plt.xticks(rotation=45)
      plt.title(f"Countplot of {col}")
      plt.tight_layout()

    plt.show()

def draw_scatterplot(df, continuous_columns):
  total_plot = len(continuous_columns)
  cols = 3
  rows = math.ceil(total_plot / cols)
  plt.figure(figsize = (4 * cols, 3 * rows))

  for i, col in enumerate(continuous_columns, 1):
    plt.subplot(rows, cols, i)
    sns.regplot(x=col, y='E', scatter=True, line_kws={"color": "red"}, data = df, color=sns.color_palette('pastel')[1])
    plt.tight_layout()
  plt.show()

def draw_boxplot(df, columns):
  total_plot = len(columns)
  cols = 3
  rows = math.ceil(total_plot / cols)
  plt.figure(figsize = (4 * cols, 3 * rows))

  for i, col in enumerate(columns, 1):
    plt.subplot(rows, cols, i)
    sns.boxplot(x = col, y = 'E', data = df, color=sns.color_palette('pastel')[1])
    plt.tight_layout()
  plt.show()

def plot_histograms_side_by_side(train_before, train_after, y_columns):
    num_cols = len(y_columns)
    fig, axes = plt.subplots(num_cols, 2, figsize=(13, 4 * num_cols))

    # 데이터가 한 개의 열인 경우, axes의 차원을 2D로 맞춰줌
    if num_cols == 1:
        axes = np.array([axes])

    for i, col in enumerate(y_columns):
        # train_before 데이터에 대한 히스토그램 (좌측)
        sns.histplot(x=col, data=train_before, ax=axes[i, 0], color=sns.color_palette("pastel")[0])
        axes[i, 0].set_title(f"Histogram of {col} (Before)")

        # train_after 데이터에 대한 히스토그램 (우측)
        sns.histplot(x=col, data=train_after, ax=axes[i, 1], color=sns.color_palette("pastel")[1])
        axes[i, 1].set_title(f"Histogram of {col} (After)")

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 X 변수와 Y 변수 간의 관계를 regplot으로 시각화하는 함수
def plot_regplots_side_by_side(train_before, train_after, x_columns, y_column):
    """
    train_before, train_after에서 X 변수들과 Y 변수 간의 관계를 regplot으로 나란히 시각화
    """
    num_features = len(x_columns)
    fig, axes = plt.subplots(nrows=num_features, ncols=2, figsize=(12, 4 * num_features))

    for i, x_col in enumerate(x_columns):
        # 🔹 Pre-Improvement 시각화
        sns.regplot(
            x=train_before[x_col], y=train_before[y_column],
            scatter_kws={'alpha': 0.3, 'color': 'blue'}, line_kws={'color': 'red'}, ax=axes[i, 0]
        )
        axes[i, 0].set_title(f"Pre-Improvement: {x_col} vs {y_column}")
        axes[i, 0].set_xlabel(x_col)
        axes[i, 0].set_ylabel(y_column)

        # 🔹 Post-Improvement 시각화
        sns.regplot(
            x=train_after[x_col], y=train_after[y_column],
            scatter_kws={'alpha': 0.3, 'color': 'orange'}, line_kws={'color': 'red'}, ax=axes[i, 1]
        )
        axes[i, 1].set_title(f"Post-Improvement: {x_col} vs {y_column}")
        axes[i, 1].set_xlabel(x_col)
        axes[i, 1].set_ylabel(y_column)

    plt.tight_layout()
    plt.show()

def plot_columns_comparison(train_before, train_after, columns=None):
    """
    train_before와 train_after 데이터셋의 모든 컬럼을 나란히 비교하는 시각화 함수
    
    Parameters:
    -----------
    train_before : pandas.DataFrame
        비교할 첫 번째 데이터셋
    train_after : pandas.DataFrame
        비교할 두 번째 데이터셋
    columns : list, optional
        시각화할 컬럼 리스트. None인 경우 모든 컬럼을 시각화
    """
    if columns is None:
        columns = train_before.columns
    
    num_cols = len(columns)
    fig, axes = plt.subplots(num_cols, 2, figsize=(15, 4 * num_cols))
    
    # 데이터가 한 개의 열인 경우, axes의 차원을 2D로 맞춰줌
    if num_cols == 1:
        axes = np.array([axes])
    
    for i, col in enumerate(columns):
        # train_before 데이터에 대한 히스토그램 (좌측)
        sns.histplot(data=train_before[col], ax=axes[i, 0], color=sns.color_palette("pastel")[0], kde=True)
        axes[i, 0].set_title(f"{col} (Before)")
        axes[i, 0].set_xlabel("")
        
        # train_after 데이터에 대한 히스토그램 (우측)
        sns.histplot(data=train_after[col], ax=axes[i, 1], color=sns.color_palette("pastel")[1], kde=True)
        axes[i, 1].set_title(f"{col} (After)")
        axes[i, 1].set_xlabel("")
        
        # y축 범위만 동일하게 설정
        y_min = min(axes[i, 0].get_ylim()[0], axes[i, 1].get_ylim()[0])
        y_max = max(axes[i, 0].get_ylim()[1], axes[i, 1].get_ylim()[1])
        axes[i, 0].set_ylim(y_min, y_max)
        axes[i, 1].set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.show()

def compare_lineplot(train_before, train_after, columns=None):
    """
    train_before와 train_after 데이터셋의 모든 컬럼을 나란히 비교하는 시각화 함수
    
    Parameters:
    -----------
    train_before : pandas.DataFrame
        비교할 첫 번째 데이터셋
    train_after : pandas.DataFrame
        비교할 두 번째 데이터셋
    columns : list, optional
        시각화할 컬럼 리스트. None인 경우 모든 컬럼을 시각화
    """
    if columns is None:
        columns = train_before.columns
    
    num_cols = len(columns)
    fig, axes = plt.subplots(num_cols, 2, figsize=(15, 4 * num_cols))
    
    # 데이터가 한 개의 열인 경우, axes의 차원을 2D로 맞춰줌
    if num_cols == 1:
        axes = np.array([axes])
    
    for i, col in enumerate(columns):
        # train_before 데이터에 대한 라인플롯 (좌측)
        sns.lineplot(x=train_before.index, y=train_before[col], ax=axes[i, 0], color=sns.color_palette("pastel")[0])
        axes[i, 0].set_title(f"{col} (Before)")
        axes[i, 0].set_xlabel("")
        
        # train_after 데이터에 대한 라인플롯 (우측)
        sns.lineplot(x=train_after.index, y=train_after[col], ax=axes[i, 1], color=sns.color_palette("pastel")[1])
        axes[i, 1].set_title(f"{col} (After)")
        axes[i, 1].set_xlabel("")
        
        # y축 범위만 동일하게 설정
        y_min = min(axes[i, 0].get_ylim()[0], axes[i, 1].get_ylim()[0])
        y_max = max(axes[i, 0].get_ylim()[1], axes[i, 1].get_ylim()[1])
        axes[i, 0].set_ylim(y_min, y_max)
        axes[i, 1].set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.show()


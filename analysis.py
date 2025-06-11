import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_operation_periods(df, label):
    # Getting_Time을 datetime으로 변환
    df["Getting_Time"] = pd.to_datetime(df["Getting_Time"])

    # 전체 기간 설정
    min_date = df["Getting_Time"].dt.date.min()
    max_date = df["Getting_Time"].dt.date.max()
    total_days = (max_date - min_date).days  # 전체 일수

    # 실제 가동된 날짜 계산
    operation_dates = sorted(df["Getting_Time"].dt.date.unique())
    operation_days = len(operation_dates)  # 가동된 일수

    # 비가동 구간 탐색
    gaps = [operation_dates[i+1] - operation_dates[i] for i in range(len(operation_dates)-1)]
    downtime_periods = [(operation_dates[i], operation_dates[i+1], gaps[i].days - 1)
                         for i in range(len(gaps)) if gaps[i].days > 1]

    # 비가동 이전 및 이후 기간 계산
    if downtime_periods:
        first_downtime_start = downtime_periods[0][0]
        last_downtime_end = downtime_periods[-1][1]
        pre_downtime_days = (first_downtime_start - min_date).days
        post_downtime_days = (max_date - last_downtime_end).days
        total_downtime_days = sum([dp[2] for dp in downtime_periods])
    else:
        first_downtime_start = last_downtime_end = None
        pre_downtime_days = post_downtime_days = total_downtime_days = 0

    # 가동 범위 설정
    first_span_start = operation_dates[0]
    first_span_end = first_downtime_start if downtime_periods else operation_dates[-1]

    second_span_start = last_downtime_end if downtime_periods else None
    second_span_end = operation_dates[-1]

    # 출력 결과
    print(f"< {label} DATA >")
    print(f"전체 일수 및 범위 = 총 {total_days}일 ({min_date} ~ {max_date})\n")

    print(f"가동 일수 및 범위 = 총 {operation_days}일 ({first_span_start} ~ {second_span_end})")
    if downtime_periods:
        print(f"- 범위 1 = 총 {(first_span_end - first_span_start).days + 1}일 ({first_span_start} ~ {first_span_end})")
        print(f"- 범위 2 = 총 {(second_span_end - second_span_start).days + 1}일 ({second_span_start} ~ {second_span_end})\n")

        print(f"생략된 일수 및 범위 = 총 {total_downtime_days}일 ({first_downtime_start + pd.Timedelta(days=1)} ~ {last_downtime_end - pd.Timedelta(days=1)})")
    else:
        print("- 비가동 구간 없음")
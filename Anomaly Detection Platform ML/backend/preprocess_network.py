import pandas as pd

def align_network_features(df: pd.DataFrame, expected_features: list) -> pd.DataFrame:
    # Add missing columns with default 0 values
    for feat in expected_features:
        if feat not in df.columns:
            df[feat] = 0
    # Drop extra columns
    df = df[expected_features]
    return df

if __name__ == "__main__":
    # Example usage
    expected_features = [
        'ifInOctets11', 'ifOutOctets11', 'ifoutDiscards11', 'ifInUcastPkts11', 'ifInNUcastPkts11',
        'ifInDiscards11', 'ifOutUcastPkts11', 'ifOutNUcastPkts11', 'tcpOutRsts', 'tcpInSegs',
        'tcpOutSegs', 'tcpPassiveOpens', 'tcpRetransSegs', 'tcpCurrEstab', 'tcpEstabResets',
        'tcp?ActiveOpens', 'udpInDatagrams', 'udpOutDatagrams', 'udpInErrors', 'udpNoPorts',
        'ipInReceives', 'ipInDelivers', 'ipOutRequests', 'ipOutDiscards', 'ipInDiscards',
        'ipForwDatagrams', 'ipOutNoRoutes', 'ipInAddrErrors', 'icmpInMsgs', 'icmpInDestUnreachs',
        'icmpOutMsgs', 'icmpOutDestUnreachs', 'icmpInEchos', 'icmpOutEchoReps', 'class'
    ]
    df = pd.read_csv("sample_network.csv")
    df_aligned = align_network_features(df, expected_features)
    df_aligned.to_csv("sample_network_aligned.csv", index=False)

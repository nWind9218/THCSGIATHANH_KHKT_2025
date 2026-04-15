CREATE TABLE user_longterm_memory (
    ip_address TEXT PRIMARY KEY,
    o_quest INT DEFAULT 0,
    c_quest INT DEFAULT 0,
    e_quest INT DEFAULT 0,
    a_quest INT DEFAULT 0,
    n_quest INT DEFAULT 0,
    o_score FLOAT DEFAULT 0.0,
    c_score FLOAT DEFAULT 0.0,
    e_score FLOAT DEFAULT 0.0,
    a_score FLOAT DEFAULT 0.0,
    n_score FLOAT DEFAULT 0.0,
    
    current_style TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION sync_user_style(target_ip TEXT)
RETURNS VOID AS $$
DECLARE
    u RECORD;
    max_score FLOAT;
BEGIN
    -- 1. Lấy dữ liệu hiện tại
    SELECT * INTO u FROM user_longterm_memory WHERE ip_address = target_ip;

    -- 2. Tính toán lại Score dựa trên luật Tương hỗ/Tương khắc
    -- Ở đây ta dùng hệ số 1.0 cho câu hỏi chính và +/- 0.2 cho các mối quan hệ
    u.o_score := u.o_quest + (u.e_quest * 0.2) - (u.c_quest * 0.1);
    u.c_score := u.c_quest + (u.a_quest * 0.2) - (u.n_quest * 0.2);
    u.e_score := u.e_quest + (u.o_quest * 0.2) - (u.n_quest * 0.2);
    u.a_score := u.a_quest + (u.c_quest * 0.2) + (u.e_quest * 0.1);
    u.n_score := u.n_quest + (u.c_quest * 0.2) - (u.e_quest * 0.1);

    -- 3. Xác định Dominant Style (Winner Takes All)
    -- Logic: Nhóm nào có điểm cao nhất sau khi tính toán sẽ được chọn
    
    -- Tìm giá trị lớn nhất trong 5 scores
    max_score := GREATEST(u.o_score, u.c_score, u.e_score, u.a_score, u.n_score);

    -- Mapping sang 8 Styles dựa trên bộ điểm (Simplified Logic)
    IF (u.o_score >= max_score * 0.8 AND u.e_score >= max_score * 0.8) THEN
        u.current_style := 'The Explorer';
    ELSIF (u.c_score >= max_score * 0.8 AND u.a_score >= max_score * 0.8) THEN
        u.current_style := 'The Achiever';
    ELSIF (u.n_score >= max_score * 0.8 AND u.a_score >= max_score * 0.8) THEN
        u.current_style := 'The Sensitive Soul';
    ELSIF (u.o_score >= max_score * 0.8 AND u.c_score < max_score * 0.5) THEN
        u.current_style := 'The Creative Dreamer';
    ELSIF (u.e_score >= max_score * 0.8 AND u.a_score >= max_score * 0.8) THEN
        u.current_style := 'The Social Star';
    ELSIF (u.c_score >= max_score * 0.8 AND u.e_score < max_score * 0.5) THEN
        u.current_style := 'The Little Analyst';
    ELSIF (u.o_score >= max_score * 0.8 AND u.a_score < max_score * 0.5) THEN
        u.current_style := 'The Rebel';
    ELSE
        u.current_style := 'The Balanced';
    END IF;

    -- 4. Lưu lại kết quả
    UPDATE user_longterm_memory SET
        o_score = u.o_score, c_score = u.c_score, e_score = u.e_score,
        a_score = u.a_score, n_score = u.n_score,
        current_style = u.current_style,
        updated_at = NOW()
    WHERE ip_address = target_ip;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_user_memory_modtime
BEFORE UPDATE ON user_longterm_memory
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();
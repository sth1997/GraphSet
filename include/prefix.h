#pragma once

class Prefix
{
public:
    Prefix();
    ~Prefix();
    void init(int input_size, const int* input_data);
    bool operator ==(const Prefix& pre) const;
    bool operator !=(const Prefix& pre) const;
    bool equal(int input_size, const int* input_data) const;
    inline int get_size() const { return size;}
    inline const int* get_data_ptr() const { return data;}
    inline int get_data(int index) const { return data[index];}
    void set_has_child(bool flag) { has_child = flag; }
    bool get_has_child() { return has_child; }
    void set_only_need_size(bool flag) { only_need_size = flag;}
    bool get_only_need_size() { return only_need_size; }

private:
    int size;
    int* data;
    bool has_child; // if other prefix depends on this prefix, used for IEP intersection optimization
    bool only_need_size;
};

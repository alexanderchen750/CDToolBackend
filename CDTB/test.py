from parser_utils import parse_input, DEFAULT_GRAMMAR

json_text = '''
{
  "order_id": "123456789",
  "customer_name": "John Doe",
  "address": "123 Main St, Anytown, USA",
  "shipping_date": "2022-01-01",
  "status": "Shipped"
}
'''

if __name__ == "__main__":
    parse_input(json_text)
    print("pass")